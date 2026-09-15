/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "runtime/state/sequenceSlots.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace trt_edgellm
{
namespace rt
{
//! Forward methods are worker-only; validToken must read immutable metadata and be thread-safe.
//! Worker-only forward boundary. Implementations complete GPU work and accept one greedy token per ready row.
class SchedulerBackend
{
public:
    virtual ~SchedulerBackend() = default;
    virtual void start() {}
    virtual void invalidate() noexcept {}
    virtual bool validToken(int32_t token) const
    {
        return token >= 0;
    }
    virtual SequenceHandle acquire(uint64_t id, std::vector<int32_t> prompt, int32_t maxOutput) = 0;
    virtual SequenceState const& state(SequenceHandle handle) const = 0;
    virtual void prefill(SequenceHandle handle) = 0;
    virtual void decode(std::array<SequenceHandle, 2> const& handles, int32_t count) = 0;
    virtual void release(SequenceHandle handle) = 0;
};

enum class SchedulerStatus
{
    kCompleted,
    kCancelled,
    kFailed
};
struct SchedulerResult
{
    SchedulerStatus status{SchedulerStatus::kFailed};
    std::vector<int32_t> tokens;
    std::exception_ptr error;
};

//! Ticket cancellation targets its submission, never a recycled physical slot.
class SchedulerTicket
{
public:
    uint64_t id() const
    {
        return mId;
    }
    void cancel() const
    {
        if (mCancelled)
        {
            mCancelled->store(true);
        }
    }
    std::shared_future<SchedulerResult> result() const
    {
        return mResult;
    }

private:
    friend class ContinuousScheduler;
    uint64_t mId{};
    std::shared_ptr<std::atomic<bool>> mCancelled;
    std::shared_future<SchedulerResult> mResult;
};

//! Fixed-size diagnostic event; callbacks run on the worker and must not block or call close().
struct SchedulerEvent
{
    enum class Kind
    {
        kAdmit,
        kPrefill,
        kDecode,
        kRelease,
        kIdle
    };
    Kind kind{};
    uint64_t request{};
    SequenceHandle handle{};
    uint64_t partner{};
    int32_t cursor{};
    int64_t microseconds{};
};

//! Internal P4 scheduler: prepared text tokens, greedy sampling and length termination only.
//! Parent runtime and stream must outlive close/destruction. Public submission and close are thread-safe.
class ContinuousScheduler
{
public:
    using Observer = std::function<void(SchedulerEvent const&)>;
    ContinuousScheduler(std::unique_ptr<SchedulerBackend> backend, size_t maxQueued = 8,
        size_t maxQueuedBytes = 256 * 1024, Observer observer = {});
    ~ContinuousScheduler();
    SchedulerTicket submit(std::vector<int32_t> const& prompt, int32_t maxOutput);
    void close();
    bool healthy() const
    {
        return mHealthy.load();
    }

private:
    struct Request
    {
        uint64_t id{};
        std::vector<int32_t> prompt;
        int32_t maxOutput{};
        std::shared_ptr<std::atomic<bool>> cancelled;
        std::promise<SchedulerResult> promise;
        SequenceHandle handle{};
    };
    void run() noexcept;
    void boundary();
    void emit(SchedulerEvent::Kind kind, Request const* request = nullptr, uint64_t partner = 0);
    void terminal(
        std::unique_ptr<Request>& request, SchedulerStatus status, bool release, std::exception_ptr error = {});
    std::unique_ptr<SchedulerBackend> mBackend;
    size_t const mMaxQueued;
    size_t const mMaxQueuedBytes;
    Observer mObserver;
    std::mutex mMutex;
    std::mutex mCloseMutex;
    std::condition_variable mWake;
    std::deque<std::unique_ptr<Request>> mQueue;
    size_t mQueuedBytes{};
    uint64_t mNextId{};
    bool mClosing{};
    std::atomic<bool> mHealthy{true};
    std::array<std::unique_ptr<Request>, 2> mActive;
    int32_t mNextPrefill{};
    std::thread mWorker;
};
} // namespace rt
} // namespace trt_edgellm
