/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/continuousScheduler.h"
#include <chrono>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
ContinuousScheduler::ContinuousScheduler(
    std::unique_ptr<SchedulerBackend> backend, size_t maxQueued, size_t maxQueuedBytes, Observer observer)
    : mBackend(std::move(backend))
    , mMaxQueued(maxQueued)
    , mMaxQueuedBytes(maxQueuedBytes)
    , mObserver(std::move(observer))
{
    if (!mBackend || !maxQueued || !maxQueuedBytes)
    {
        throw std::invalid_argument("Scheduler requires a backend and positive queue bounds");
    }
    mWorker = std::thread(&ContinuousScheduler::run, this);
}
ContinuousScheduler::~ContinuousScheduler()
{
    close();
}

SchedulerTicket ContinuousScheduler::submit(std::vector<int32_t> const& prompt, int32_t maxOutput)
{
    if (prompt.empty() || prompt.size() > 6144 || maxOutput <= 0
        || maxOutput > 8192 - static_cast<int32_t>(prompt.size()))
    {
        throw std::invalid_argument("P4 request exceeds qualified token bounds");
    }
    for (auto token : prompt)
    {
        if (!mBackend->validToken(token))
        {
            throw std::invalid_argument("Token outside supported vocabulary");
        }
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || !mHealthy.load())
    {
        throw std::runtime_error("Scheduler closed or failed");
    }
    size_t const bytes = prompt.size() * sizeof(int32_t);
    if (mQueue.size() >= mMaxQueued || bytes > mMaxQueuedBytes - mQueuedBytes)
    {
        throw std::runtime_error("Scheduler queue full");
    }
    if (mNextId == std::numeric_limits<uint64_t>::max())
    {
        throw std::overflow_error("Scheduler ticket IDs exhausted");
    }
    auto request = std::make_unique<Request>();
    request->id = ++mNextId;
    request->prompt = prompt;
    request->maxOutput = maxOutput;
    request->cancelled = std::make_shared<std::atomic<bool>>(false);
    SchedulerTicket ticket;
    ticket.mId = request->id;
    ticket.mCancelled = request->cancelled;
    ticket.mResult = request->promise.get_future().share();
    mQueue.push_back(std::move(request));
    mQueuedBytes += bytes;
    mWake.notify_one();
    return ticket;
}
void ContinuousScheduler::close()
{
    std::lock_guard<std::mutex> joinLock(mCloseMutex);
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mClosing = true;
    }
    mWake.notify_one();
    if (mWorker.joinable())
    {
        mWorker.join();
    }
}
void ContinuousScheduler::emit(SchedulerEvent::Kind kind, Request const* request, uint64_t partner)
{
    if (!mObserver)
    {
        return;
    }
    SchedulerEvent event;
    event.kind = kind;
    event.partner = partner;
    event.microseconds
        = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
              .count();
    if (request)
    {
        event.request = request->id;
        event.handle = request->handle;
        event.cursor = mBackend->state(request->handle).promptCursor();
    }
    mObserver(event);
}
void ContinuousScheduler::terminal(
    std::unique_ptr<Request>& request, SchedulerStatus status, bool release, std::exception_ptr error)
{
    SchedulerResult result;
    result.status = status;
    result.error = error;
    if (release)
    {
        result.tokens = mBackend->state(request->handle).output();
        emit(SchedulerEvent::Kind::kRelease, request.get());
        mBackend->release(request->handle);
    }
    request->promise.set_value(std::move(result));
    request.reset();
}
void ContinuousScheduler::boundary()
{
    // No GPU work is outstanding here. Reclaim before admission, including between decode and prefill.
    for (auto& request : mActive)
    {
        if (!request)
        {
            continue;
        }
        if (request->cancelled->load())
        {
            terminal(request, SchedulerStatus::kCancelled, true);
        }
        else if (mBackend->state(request->handle).phase() == SequencePhase::kFinished)
        {
            terminal(request, SchedulerStatus::kCompleted, true);
        }
    }
    for (auto& request : mActive)
    {
        size_t examined = 0;
        while (!request && examined++ < mMaxQueued)
        {
            {
                std::lock_guard<std::mutex> lock(mMutex);
                if (mClosing || mQueue.empty())
                {
                    break;
                }
                request = std::move(mQueue.front());
                mQueue.pop_front();
                mQueuedBytes -= request->prompt.size() * sizeof(int32_t);
            }
            if (request->cancelled->load())
            {
                terminal(request, SchedulerStatus::kCancelled, false);
                continue;
            }
            request->handle = mBackend->acquire(request->id, std::move(request->prompt), request->maxOutput);
            emit(SchedulerEvent::Kind::kAdmit, request.get());
        }
    }
}
void ContinuousScheduler::run() noexcept
{
    try
    {
        mBackend->start();
        while (true)
        {
            {
                std::unique_lock<std::mutex> lock(mMutex);
                if (!mClosing && mQueue.empty() && !mActive[0] && !mActive[1])
                {
                    lock.unlock();
                    emit(SchedulerEvent::Kind::kIdle);
                    lock.lock();
                    mWake.wait(lock, [&] { return mClosing || !mQueue.empty(); });
                }
                if (mClosing)
                {
                    break;
                }
            }
            boundary();
            std::array<SequenceHandle, 2> handles{};
            std::array<Request*, 2> decoding{};
            int32_t count = 0;
            for (auto& request : mActive)
            {
                if (request && mBackend->state(request->handle).phase() == SequencePhase::kDecode)
                {
                    handles[count] = request->handle;
                    decoding[count++] = request.get();
                }
            }
            if (count == 2 && handles[0].slot > handles[1].slot)
            {
                std::swap(handles[0], handles[1]);
                std::swap(decoding[0], decoding[1]);
            }
            if (count)
            {
                mBackend->decode(handles, count);
                for (int32_t i = 0; i < count; ++i)
                {
                    emit(SchedulerEvent::Kind::kDecode, decoding[i], count == 2 ? decoding[1 - i]->id : 0);
                }
            }
            boundary();
            for (int32_t offset = 0; offset < 2; ++offset)
            {
                int32_t const slot = (mNextPrefill + offset) % 2;
                auto& request = mActive[slot];
                if (request && mBackend->state(request->handle).phase() == SequencePhase::kPrefill)
                {
                    mBackend->prefill(request->handle);
                    emit(SchedulerEvent::Kind::kPrefill, request.get());
                    mNextPrefill = (slot + 1) % 2;
                    break;
                }
            }
            boundary();
        }
        for (auto& request : mActive)
        {
            if (request)
            {
                terminal(request, SchedulerStatus::kCancelled, true);
            }
        }
    }
    catch (...)
    {
        mHealthy.store(false);
        mBackend->invalidate();
        auto const error = std::current_exception();
        for (auto& request : mActive)
        {
            if (request)
            {
                terminal(request, SchedulerStatus::kFailed, false, error);
            }
        }
    }
    std::lock_guard<std::mutex> lock(mMutex);
    mClosing = true;
    for (auto& request : mQueue)
    {
        terminal(request, mHealthy.load() ? SchedulerStatus::kCancelled : SchedulerStatus::kFailed, false,
            mHealthy.load() ? std::exception_ptr{} : std::make_exception_ptr(std::runtime_error("Worker failed")));
    }
    mQueue.clear();
    mQueuedBytes = 0;
}
} // namespace rt
} // namespace trt_edgellm
