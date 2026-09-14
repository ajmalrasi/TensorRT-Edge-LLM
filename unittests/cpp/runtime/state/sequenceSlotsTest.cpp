/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/state/sequenceSlots.h"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
TEST(SequenceSlots, RejectsInvalidCapacity)
{
    EXPECT_THROW(SequenceSlots(0, 8, 16), std::invalid_argument);
    EXPECT_THROW(SequenceSlots(2, 16, 8), std::invalid_argument);
    SequenceSlots slots(2, 8, 16);
    EXPECT_THROW(slots.acquire(1, {}, {}), std::invalid_argument);
    EXPECT_THROW(slots.acquire(1, {1}, {}), std::invalid_argument);
}

TEST(SequenceSlots, AdmissionIsBoundedAndIdsAreUnique)
{
    SequenceSlots slots(2, 8, 16);
    SequenceOptions options;
    options.maxOutputTokens = 4;
    auto a = slots.acquire(1, {1, 2}, options);
    EXPECT_THROW(slots.acquire(1, {3}, options), std::invalid_argument);
    auto b = slots.acquire(2, {3}, options);
    EXPECT_EQ(a.slot, 0);
    EXPECT_EQ(b.slot, 1);
    EXPECT_THROW(slots.acquire(3, {3}, options), std::runtime_error);
}

TEST(SequenceSlots, RejectsStaleAndForeignHandles)
{
    SequenceSlots slots(2, 8, 16);
    SequenceSlots other(2, 8, 16);
    SequenceOptions options;
    options.maxOutputTokens = 4;
    auto a = slots.acquire(1, {1}, options);
    EXPECT_THROW(other.get(a), std::invalid_argument);
    slots.release(a);
    EXPECT_THROW(slots.get(a), std::invalid_argument);
    auto b = slots.acquire(2, {2}, options);
    EXPECT_EQ(a.slot, b.slot);
    EXPECT_NE(a.generation, b.generation);
    EXPECT_THROW(slots.finish(a), std::invalid_argument);
    EXPECT_EQ(slots.get(b).requestId(), 2U);
}

TEST(SequenceSlots, SeparatesSampledAndCommittedTokens)
{
    SequenceSlots slots(2, 8, 16);
    SequenceOptions options;
    options.maxOutputTokens = 2;
    auto a = slots.acquire(1, {1, 2, 3}, options);
    slots.commitPrompt(a, 2);
    EXPECT_EQ(slots.get(a).promptCursor(), 2);
    EXPECT_THROW(slots.acceptToken(a, 4), std::logic_error);
    slots.commitPrompt(a, 1);
    EXPECT_EQ(slots.get(a).phase(), SequencePhase::kAwaitingSample);
    slots.acceptToken(a, 4, 7);
    EXPECT_EQ(slots.get(a).committedTokens(), 3);
    EXPECT_EQ(slots.get(a).randomCounter(), 7U);
    slots.commitDecode(a);
    EXPECT_EQ(slots.get(a).committedTokens(), 4);
    EXPECT_THROW(slots.commitDecode(a), std::logic_error);
    slots.acceptToken(a, 5);
    EXPECT_EQ(slots.get(a).phase(), SequencePhase::kFinished);
    EXPECT_EQ(slots.get(a).output().size(), 2U);
    EXPECT_EQ(slots.get(a).committedTokens(), 4);
    EXPECT_THROW(slots.acceptToken(a, 6), std::logic_error);
}

TEST(SequenceSlots, OneTokenOutputNeedsNoDecodeForward)
{
    SequenceSlots slots(2, 8, 16);
    SequenceOptions options;
    options.maxOutputTokens = 1;
    auto a = slots.acquire(1, {1, 2}, options);
    slots.commitPrompt(a, 2);
    slots.acceptToken(a, 3);
    EXPECT_EQ(slots.get(a).phase(), SequencePhase::kFinished);
    EXPECT_EQ(slots.get(a).committedTokens(), 2);
    EXPECT_THROW(slots.commitDecode(a), std::logic_error);
}

TEST(SequenceSlots, RequestsKeepIndependentOptionsAndHeadroom)
{
    SequenceSlots slots(2, 12, 16);
    SequenceOptions first;
    first.maxOutputTokens = 2;
    first.temperature = 0.0F;
    first.stopStrings = {"END"};
    first.logitBias[7] = -2.0F;
    SequenceOptions second;
    second.maxOutputTokens = 8;
    second.seed = 123;
    auto a = slots.acquire(1, std::vector<int32_t>(12, 1), first);
    auto b = slots.acquire(2, {1}, second);
    slots.commitPrompt(a, 12);
    slots.acceptToken(a, 2);
    slots.finish(a);
    slots.release(a);
    EXPECT_EQ(slots.get(b).options().maxOutputTokens, 8);
    EXPECT_EQ(slots.get(b).options().seed, 123U);
    EXPECT_TRUE(slots.get(b).options().stopStrings.empty());
    EXPECT_EQ(slots.get(b).committedTokens(), 0);
}

TEST(SequenceSlots, InvalidTransitionsDoNotMutateState)
{
    SequenceSlots slots(2, 8, 16);
    SequenceOptions options;
    options.maxOutputTokens = 2;
    auto a = slots.acquire(1, {1, 2}, options);
    EXPECT_THROW(slots.commitPrompt(a, 3), std::logic_error);
    EXPECT_THROW(slots.commitPrompt(a, 0), std::logic_error);
    EXPECT_THROW(slots.commitDecode(a), std::logic_error);
    EXPECT_EQ(slots.get(a).promptCursor(), 0);
    slots.commitPrompt(a, 2);
    EXPECT_THROW(slots.commitPrompt(a, 1), std::logic_error);
    EXPECT_THROW(slots.acceptToken(a, -1), std::logic_error);
    EXPECT_TRUE(slots.get(a).output().empty());
}

TEST(SequenceSlots, DetectsRandomCounterOverflow)
{
    SequenceSlots slots(2, 8, 16);
    SequenceOptions options;
    options.maxOutputTokens = 3;
    auto a = slots.acquire(1, {1}, options);
    slots.commitPrompt(a, 1);
    slots.acceptToken(a, 2, std::numeric_limits<uint64_t>::max());
    slots.commitDecode(a);
    EXPECT_THROW(slots.acceptToken(a, 3, 1), std::logic_error);
    EXPECT_EQ(slots.get(a).output().size(), 1U);
}

TEST(SequenceSlots, ReuseDoesNotCarryPromptOutputOrOptions)
{
    SequenceSlots slots(1, 8, 16);
    SequenceOptions options;
    options.maxOutputTokens = 1;
    for (uint64_t request = 1; request <= 100; ++request)
    {
        auto handle = slots.acquire(request, {static_cast<int32_t>(request)}, options);
        EXPECT_EQ(slots.get(handle).committedTokens(), 0);
        EXPECT_EQ(slots.get(handle).randomCounter(), 0U);
        EXPECT_TRUE(slots.get(handle).output().empty());
        slots.commitPrompt(handle, 1);
        slots.acceptToken(handle, 1);
        slots.release(handle);
    }
}
} // namespace rt
} // namespace trt_edgellm
