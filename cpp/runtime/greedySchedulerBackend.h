/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "runtime/continuousScheduler.h"
#include "runtime/sequenceStepRuntime.h"

namespace trt_edgellm
{
namespace rt
{
//! P4-only greedy adapter. Borrowed runtime and explicit stream outlive this exclusive lease.
class GreedySchedulerBackend : public SchedulerBackend
{
public:
    GreedySchedulerBackend(LLMRankRuntime& runtime, cudaStream_t stream, int32_t vocabularySize);
    void start() override;
    void invalidate() noexcept override
    {
        mSteps.poison();
    }
    bool validToken(int32_t token) const override
    {
        return token >= 0 && token < mVocabulary;
    }
    SequenceHandle acquire(uint64_t id, std::vector<int32_t> prompt, int32_t maxOutput) override;
    SequenceState const& state(SequenceHandle handle) const override;
    void prefill(SequenceHandle handle) override;
    void decode(std::array<SequenceHandle, 2> const& handles, int32_t count) override;
    void release(SequenceHandle handle) override;

private:
    void sample(Tensor const& logits, std::array<SequenceHandle, 2> const& handles, int32_t count);
    SequenceStepRuntime mSteps;
    cudaStream_t mStream;
    int32_t mVocabulary;
    int mDevice{};
    Tensor mHostLogits;
};
} // namespace rt
} // namespace trt_edgellm
