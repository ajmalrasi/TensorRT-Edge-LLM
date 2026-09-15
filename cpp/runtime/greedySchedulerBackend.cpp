/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/greedySchedulerBackend.h"
#include "common/checkMacros.h"
#include <cmath>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
GreedySchedulerBackend::GreedySchedulerBackend(LLMRankRuntime& runtime, cudaStream_t stream, int32_t vocabularySize)
    : mSteps(runtime, stream)
    , mStream(stream)
    , mVocabulary(vocabularySize)
    , mHostLogits({2, vocabularySize}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT)
{
    CUDA_CHECK(cudaGetDevice(&mDevice));
}
void GreedySchedulerBackend::start()
{
    CUDA_CHECK(cudaSetDevice(mDevice));
}
SequenceHandle GreedySchedulerBackend::acquire(uint64_t id, std::vector<int32_t> prompt, int32_t maxOutput)
{
    for (auto token : prompt)
    {
        if (token < 0 || token >= mVocabulary)
        {
            throw std::invalid_argument("Token outside vocabulary");
        }
    }
    SequenceOptions options;
    options.maxOutputTokens = maxOutput;
    options.temperature = 0.0F;
    return mSteps.acquire(id, std::move(prompt), std::move(options));
}
SequenceState const& GreedySchedulerBackend::state(SequenceHandle handle) const
{
    return mSteps.state(handle);
}
void GreedySchedulerBackend::sample(Tensor const& logits, std::array<SequenceHandle, 2> const& handles, int32_t count)
{
    if (logits.getDataType() != nvinfer1::DataType::kFLOAT || logits.getShape()[0] != count
        || logits.getShape()[1] != mVocabulary)
    {
        throw std::runtime_error("Unexpected scheduler logits");
    }
    CUDA_CHECK(cudaMemcpyAsync(mHostLogits.rawPointer(), logits.rawPointer(), count * mVocabulary * sizeof(float),
        cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    auto const* data = mHostLogits.dataPointer<float>();
    for (int32_t row = 0; row < count; ++row)
    {
        auto const* values = data + row * mVocabulary;
        int32_t best = 0;
        for (int32_t token = 0; token < mVocabulary; ++token)
        {
            if (!std::isfinite(values[token]))
            {
                throw std::runtime_error("Nonfinite scheduler logits");
            }
            if (values[token] > values[best])
            {
                best = token;
            }
        }
        mSteps.acceptToken(handles[row], best);
    }
}
void GreedySchedulerBackend::prefill(SequenceHandle handle)
{
    auto const& logits = mSteps.beginPrefillChunk(handle);
    mSteps.completeStep();
    if (state(handle).phase() == SequencePhase::kAwaitingSample)
    {
        sample(logits, {handle, {}}, 1);
    }
}
void GreedySchedulerBackend::decode(std::array<SequenceHandle, 2> const& handles, int32_t count)
{
    auto const& logits = mSteps.beginDecode(handles, count);
    mSteps.completeStep();
    sample(logits, handles, count);
}
void GreedySchedulerBackend::release(SequenceHandle handle)
{
    mSteps.finish(handle);
    mSteps.release(handle);
}
} // namespace rt
} // namespace trt_edgellm
