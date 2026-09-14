/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/bindingNames.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/sequenceStepRuntime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{
void require(bool condition, std::string const& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

int32_t greedy(std::vector<float> const& logits)
{
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

bool compare(std::string const& label, std::vector<float> const& reference, std::vector<float> const& candidate,
    bool exact = false)
{
    require(reference.size() == candidate.size(), "Logit count mismatch");
    double squaredError = 0.0;
    double squaredReference = 0.0;
    float maxError = 0.0F;
    bool finite = true;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        finite = finite && std::isfinite(reference[i]) && std::isfinite(candidate[i]);
        double const error = static_cast<double>(candidate[i]) - reference[i];
        squaredError += error * error;
        squaredReference += static_cast<double>(reference[i]) * reference[i];
        maxError = std::max(maxError, static_cast<float>(std::abs(error)));
    }
    double const relativeL2 = std::sqrt(squaredError / std::max(squaredReference, 1.0e-30));
    bool const sameGreedy = greedy(reference) == greedy(candidate);
    // Preliminary FP16 correctness screen, not a model-quality acceptance criterion.
    bool const passed = finite && sameGreedy && (exact ? maxError == 0.0F : maxError <= 0.1F && relativeL2 <= 0.005);
    std::cout << "COMPARE " << label << " max_abs=" << maxError << " relative_l2=" << relativeL2
              << " reference_greedy=" << greedy(reference) << " candidate_greedy=" << greedy(candidate)
              << " passed=" << passed << std::endl;
    return passed;
}
} // namespace

//! Model-dependent, eager-only feasibility probe; deliberately not a serving API.
class ContinuousBatchingProbe
{
public:
    ContinuousBatchingProbe(LLMRankRuntime& runtime, cudaStream_t stream)
        : mRuntime(runtime)
        , mStream(stream)
        , mOriginalMap(runtime.mBaseTensorMap)
        , mHostLengths({2}, DeviceType::kCPU, nvinfer1::DataType::kINT32)
        , mHostLogits({2, runtime.mDeployment.base.outputVocabSize}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT)
        , mScratch({kCOPY_BYTES}, DeviceType::kCPU, nvinfer1::DataType::kUINT8)
    {
        auto const& config = runtime.mDeployment.base;
        require(runtime.mMaxRuntimeBatchSize == 2 && !runtime.hasDraftModel(), "Requires two-slot vanilla engine");
        require(config.maxSupportedLoraRank == 0 && !config.useVisionBidirectionalAttention
                && config.reducedVocabSize == 0 && !config.isDiffusionBackbone,
            "Unsupported probe configuration");
        require(runtime.mPipelineIO->outputLogits.getDataType() == nvinfer1::DataType::kFLOAT,
            "Probe requires FP32 logits");
        require(runtime.mSharedResources->kvPageTables[0]->isIdentity(), "Probe requires identity KV pages");
        if (config.ropeConfig.type == RopeType::kMRope)
        {
            auto& rope = runtime.mPipelineIO->mropeCosSin;
            require(rope.reshape({2, config.maxKVCacheCapacity, config.rotaryDim}), "RoPE shape");
            kernel::initializeTextOnlyMRopeCosSin(rope.dataPointer<float>(), config.ropeConfig.rotaryTheta,
                config.rotaryDim, config.maxKVCacheCapacity, 2, stream);
        }
        auto& mamba = cache().getMambaCacheManager();
        require(mamba.numLayers() > 0, "Probe requires hybrid recurrent state");
        for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
        {
            addBinding(binding_names::formatRecurrentStateName(layer, true));
            addBinding(binding_names::formatRecurrentStateName(layer, false));
            addBinding(binding_names::formatConvStateName(layer, true));
            addBinding(binding_names::formatConvStateName(layer, false));
        }
        addBinding(binding_names::kKVPageTable);
        reset(0);
        reset(1);
        memory("initialized");
    }

    ~ContinuousBatchingProbe()
    {
        mRuntime.mBaseTensorMap = mOriginalMap;
    }

    //! Reset only the selected sequence's recurrent state and logical endpoint.
    void reset(int32_t slot)
    {
        require(slot >= 0 && slot < 2, "Invalid slot");
        mRuntime.zeroRecurrentStates(slot, mStream);
        mCommitted[slot] = 0;
    }

    //! Execute one contiguous physical-slot view using existing native primitives.
    std::vector<std::vector<float>> step(
        int32_t firstSlot, std::vector<std::vector<int32_t>> const& tokens, bool decode = false)
    {
        int32_t const count = static_cast<int32_t>(tokens.size());
        require(count > 0 && count <= 2 && firstSlot >= 0 && firstSlot + count <= 2, "Invalid selected slots");
        require(decode || count == 1, "Probe prefill selects one slot at a time");
        for (auto const& row : tokens)
        {
            require(!row.empty() && (!decode || row.size() == 1), "Invalid step token count");
        }
        select(firstSlot, count);
        DecodingInferenceContext context;
        context.initialize(count, 8, std::nullopt, {}, "", mStream);
        context.temperature = 0.0F;
        context.topK = 1;
        context.tokenIds = tokens;
        for (int32_t i = 0; i < count; ++i)
        {
            require(mCommitted[firstSlot + i] + static_cast<int32_t>(tokens[i].size())
                    <= mRuntime.mDeployment.base.maxKVCacheCapacity,
                "KV endpoint exceeds engine capacity");
            context.effectivePrefillLengths[i] = static_cast<int32_t>(tokens[i].size());
        }
        // Nonempty-cache S=1 is decoded by the attention plugin and needs absolute context lengths.
        bool const promptTail = !decode && tokens[0].size() == 1 && mCommitted[firstSlot] > 0;
        bool const executeDecode = decode || promptTail;
        std::cout << "STEP slot=" << firstSlot << " batch=" << count << " profile=" << (executeDecode ? 1 : 0)
                  << " logical_decode=" << decode << " prompt_tail=" << promptTail << " start=" << mCommitted[firstSlot]
                  << " tokens=" << tokens[0].size() << std::endl;
        bool const success = executeDecode ? mRuntime.mDecoderRegistry->cachePrimingStrategy().decodeStep(context)
                                           : mRuntime.runBaseModelPrefill(context, nullptr, false);
        require(success, "Native execution failed");
        int32_t const vocab = mRuntime.mDeployment.base.outputVocabSize;
        CUDA_CHECK(cudaMemcpyAsync(mHostLogits.rawPointer(), mRuntime.mPipelineIO->outputLogits.rawPointer(),
            static_cast<size_t>(count) * vocab * sizeof(float), cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mHostLengths.rawPointer(), cache().getKVCacheLengths().rawPointer(),
            count * sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        std::vector<std::vector<float>> results;
        for (int32_t i = 0; i < count; ++i)
        {
            mCommitted[firstSlot + i] += static_cast<int32_t>(tokens[i].size());
            require(mHostLengths.dataPointer<int32_t>()[i] == mCommitted[firstSlot + i], "Incorrect committed length");
            auto const* begin = mHostLogits.dataPointer<float>() + i * vocab;
            results.emplace_back(begin, begin + vocab);
        }
        return results;
    }

    //! Snapshot all recurrent/conv bytes and the materialized attention-KV prefix.
    void observeEndpoint(int32_t slot, int32_t length)
    {
        mCommitted.at(slot) = length;
    }

    //! Capture sampler output scratch to detect accidental sampling by forward-only APIs.
    std::vector<std::byte> samplingBytes()
    {
        auto& indices = mRuntime.mSamplingIndices;
        std::vector<std::byte> result(indices.getMemoryCapacity());
        copyToHost(indices.rawPointer(), result.data(), result.size());
        return result;
    }

    //! Snapshot all recurrent/conv bytes and the materialized attention-KV prefix.
    void snapshot(int32_t slot)
    {
        mSnapshots.clear();
        mSnapshotSlot = slot;
        mSnapshotLength = mCommitted[slot];
        auto capture = [&](void* pointer, size_t bytes) {
            Snapshot entry{pointer, std::vector<std::byte>(bytes)};
            copyToHost(pointer, entry.bytes.data(), bytes);
            mSnapshots.push_back(std::move(entry));
        };
        auto& mamba = cache().getMambaCacheManager();
        for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
        {
            for (Tensor* tensor : {&mamba.getRecurrentState(layer), &mamba.getConvState(layer)})
            {
                size_t const bytes = tensor->getMemoryCapacity() / 2;
                capture(static_cast<std::byte*>(tensor->rawPointer()) + slot * bytes, bytes);
            }
        }
        auto& kv = cache().getKVCacheManager();
        for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
        {
            auto& tensor = kv.getCombinedKVCache(layer);
            auto const shape = tensor.getShape();
            size_t const tokenBytes = shape[3] * shape[4] * sizeof(half);
            require(tensor.getDataType() == nvinfer1::DataType::kHALF, "Probe requires FP16 KV");
            int64_t const pagesPerSlot = (mRuntime.mDeployment.base.maxKVCacheCapacity + shape[2] - 1) / shape[2];
            require(shape[1] == 2 * pagesPerSlot, "Probe requires a two-slot pool with no surplus pages");
            size_t const halfBytes = tensor.getMemoryCapacity() / 2;
            size_t const slotBytes = halfBytes / 2;
            require(slotBytes / tokenBytes >= static_cast<size_t>(mCommitted[slot]), "KV snapshot range");
            for (int32_t plane = 0; plane < 2; ++plane)
            {
                capture(static_cast<std::byte*>(tensor.rawPointer()) + plane * halfBytes + slot * slotBytes,
                    mCommitted[slot] * tokenBytes);
            }
        }
        auto& table = mRuntime.mSharedResources->kvPageTables[0]->kernelView();
        size_t const tableBytes = table.getMemoryCapacity() / 2;
        capture(static_cast<std::byte*>(table.rawPointer()) + slot * tableBytes, tableBytes);
    }

    //! Require byte-for-byte preservation of the inactive request's live state.
    void verifySnapshot()
    {
        require(mSnapshotSlot >= 0 && mCommitted[mSnapshotSlot] == mSnapshotLength, "Inactive endpoint changed");
        size_t totalBytes = 0;
        for (auto const& entry : mSnapshots)
        {
            for (size_t offset = 0; offset < entry.bytes.size(); offset += kCOPY_BYTES)
            {
                size_t const bytes = std::min(static_cast<size_t>(kCOPY_BYTES), entry.bytes.size() - offset);
                CUDA_CHECK(cudaMemcpyAsync(mScratch.rawPointer(), static_cast<std::byte*>(entry.pointer) + offset,
                    bytes, cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
                require(std::memcmp(mScratch.rawPointer(), entry.bytes.data() + offset, bytes) == 0,
                    "Inactive state modified");
                totalBytes += bytes;
            }
        }
        std::cout << "ISOLATION slot=" << mSnapshotSlot << " exact_bytes=" << totalBytes << " passed=1" << std::endl;
    }

    //! Record device allocator availability, not process RSS or exclusive GPU use.
    void memory(std::string const& label)
    {
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        std::cout << "MEMORY " << label << " cuda_free=" << freeBytes << " cuda_total=" << totalBytes << std::endl;
    }

private:
    struct Binding
    {
        Tensor* original;
        Coords shape;
    };
    struct Snapshot
    {
        void* pointer;
        std::vector<std::byte> bytes;
    };
    static constexpr int64_t kCOPY_BYTES = 1024 * 1024;

    HybridCacheManager& cache()
    {
        return *mRuntime.mSharedResources->cacheManagers[0];
    }

    void addBinding(std::string const& name)
    {
        auto* tensor = mOriginalMap.get(name);
        require(tensor != nullptr && tensor->getShape()[0] == 2, "Invalid physical binding: " + name);
        mBindings.emplace(name, Binding{tensor, tensor->getShape()});
    }

    void select(int32_t firstSlot, int32_t count)
    {
        for (auto const& [name, binding] : mBindings)
        {
            auto shape = binding.shape;
            size_t const rowBytes = binding.original->getMemoryCapacity() / 2;
            shape[0] = count;
            auto* pointer = static_cast<std::byte*>(binding.original->rawPointer()) + firstSlot * rowBytes;
            mViews[name] = Tensor(pointer, shape, DeviceType::kGPU, binding.original->getDataType());
            mRuntime.mBaseTensorMap.set(name, mViews.at(name));
        }
        require(mHostLengths.reshape({count}), "Selected lengths shape");
        for (int32_t i = 0; i < count; ++i)
        {
            mHostLengths.dataPointer<int32_t>()[i] = mCommitted[firstSlot + i];
        }
        // Cache-manager lengths are execution scratch here; canonical endpoints stay slot-owned.
        Tensor lengths(mHostLengths.rawPointer(), {count}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        cache().resetForNewSequences(lengths, mStream);
    }

    void copyToHost(void* source, std::byte* destination, size_t total)
    {
        for (size_t offset = 0; offset < total; offset += kCOPY_BYTES)
        {
            size_t const bytes = std::min(static_cast<size_t>(kCOPY_BYTES), total - offset);
            CUDA_CHECK(cudaMemcpyAsync(mScratch.rawPointer(), static_cast<std::byte*>(source) + offset, bytes,
                cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            std::memcpy(destination + offset, mScratch.rawPointer(), bytes);
        }
    }

    LLMRankRuntime& mRuntime;
    cudaStream_t mStream;
    TensorMap mOriginalMap;
    Tensor mHostLengths;
    Tensor mHostLogits;
    Tensor mScratch;
    std::array<int32_t, 2> mCommitted{};
    std::map<std::string, Binding> mBindings;
    std::map<std::string, Tensor> mViews;
    std::vector<Snapshot> mSnapshots;
    int32_t mSnapshotSlot{-1};
    int32_t mSnapshotLength{};
};

//! Run bounded synthetic continuation and staggered slot-isolation fixtures.
bool runProbe(LLMRankRuntime& runtime, tokenizer::Tokenizer& tokenizer, cudaStream_t stream)
{
    ContinuousBatchingProbe probe(runtime, stream);
    auto const vocabulary = tokenizer.encode("The red fox walks beside the blue river. Count one two three four. ");
    require(!vocabulary.empty(), "Empty synthetic fixture");
    std::vector<int32_t> prompt;
    for (int32_t i = 0; i < 129; ++i)
    {
        prompt.push_back(static_cast<int32_t>(vocabulary[i % vocabulary.size()]));
    }
    bool passed = true;
    for (int32_t length : {1, 3, 4, 63, 64, 65, 127, 128, 129})
    {
        std::vector<int32_t> const input(prompt.begin(), prompt.begin() + length);
        probe.reset(0);
        auto const reference = probe.step(0, {input})[0];
        int32_t const forcedToken = greedy(reference);
        auto const referenceDecode = probe.step(0, {{forcedToken}}, true)[0];
        probe.reset(1);
        std::vector<float> candidate;
        for (int32_t offset = 0; offset < length;)
        {
            int32_t const end = std::min(offset + 64, length);
            candidate = probe.step(1, {{input.begin() + offset, input.begin() + end}})[0];
            offset = end;
        }
        passed = compare("chunk_prefill_" + std::to_string(length), reference, candidate, length <= 64) && passed;
        auto const candidateDecode = probe.step(1, {{forcedToken}}, true)[0];
        passed = compare("chunk_decode_" + std::to_string(length), referenceDecode, candidateDecode, length <= 64)
            && passed;
    }

    probe.reset(0);
    auto const firstA = probe.step(0, {prompt})[0];
    auto const tokenA = greedy(firstA);
    auto const nextA = probe.step(0, {{tokenA}}, true)[0];
    probe.reset(1);
    std::vector<int32_t> promptB(prompt.rbegin(), prompt.rend());
    auto const firstB = probe.step(1, {promptB})[0];
    auto const tokenB = greedy(firstB);
    auto const nextB = probe.step(1, {{tokenB}}, true)[0];

    probe.reset(0);
    passed = compare("A_reuse", firstA, probe.step(0, {prompt})[0], true) && passed;
    probe.snapshot(0);
    probe.reset(1);
    probe.step(1, {{promptB.begin(), promptB.begin() + 64}});
    probe.verifySnapshot();
    probe.snapshot(1);
    passed = compare("A_decode_during_B_prefill", nextA, probe.step(0, {{tokenA}}, true)[0], true) && passed;
    probe.verifySnapshot();
    probe.snapshot(0);
    probe.step(1, {{promptB.begin() + 64, promptB.begin() + 128}});
    probe.verifySnapshot();
    auto const chunkB = probe.step(1, {{promptB.back()}})[0];
    probe.verifySnapshot();
    passed = compare("B_one_token_tail", firstB, chunkB) && passed;
    probe.snapshot(0);
    passed = compare("B_decode_after_A", nextB, probe.step(1, {{tokenB}}, true)[0]) && passed;
    probe.verifySnapshot();

    auto const secondTokenA = greedy(nextA);
    auto const secondTokenB = greedy(nextB);
    probe.reset(0);
    probe.reset(1);
    probe.step(0, {prompt});
    probe.step(1, {promptB});
    probe.step(0, {{tokenA}}, true);
    probe.step(1, {{tokenB}}, true);
    auto const secondA = probe.step(0, {{secondTokenA}}, true)[0];
    auto const secondB = probe.step(1, {{secondTokenB}}, true)[0];
    probe.reset(0);
    probe.reset(1);
    probe.step(0, {prompt});
    probe.step(1, {promptB});
    probe.step(0, {{tokenA}}, true);
    probe.step(1, {{tokenB}}, true);
    auto const batched = probe.step(0, {{secondTokenA}, {secondTokenB}}, true);
    passed = compare("two_slot_decode_A", secondA, batched[0]) && passed;
    passed = compare("two_slot_decode_B", secondB, batched[1]) && passed;
    probe.memory("completed");
    return passed;
}
} // namespace rt
} // namespace trt_edgellm

namespace trt_edgellm
{
namespace rt
{
//! Validate the production step interfaces against the independent P1/legacy path.
bool runStepTests(LLMRankRuntime& runtime, tokenizer::Tokenizer& tokenizer, cudaStream_t stream)
{
    auto const words = tokenizer.encode("A red fox crosses a blue river. One two three four. ");
    require(!words.empty(), "Missing fixture tokens");
    std::vector<int32_t> promptA;
    std::vector<int32_t> promptB;
    for (int32_t i = 0; i < 129; ++i)
    {
        promptB.push_back(words[i % words.size()]);
        if (i < 65)
        {
            promptA.push_back(words[i % words.size()]);
        }
    }
    std::vector<float> firstA, firstB, nextA, nextB, chunkFirstB, chunkNextB;
    {
        ContinuousBatchingProbe reference(runtime, stream);
        firstA = reference.step(0, {promptA})[0];
        firstB = reference.step(1, {promptB})[0];
        nextA = reference.step(0, {{greedy(firstA)}}, true)[0];
        nextB = reference.step(1, {{greedy(firstB)}}, true)[0];
        reference.reset(1);
        reference.step(1, {{promptB.begin(), promptB.begin() + 64}});
        reference.step(1, {{promptB.begin() + 64, promptB.begin() + 128}});
        chunkFirstB = reference.step(1, {{promptB.back()}})[0];
        chunkNextB = reference.step(1, {{greedy(firstB)}}, true)[0];
        bool const baselineQuality = compare("P1_decode_tail_quality_diagnostic", nextB, chunkNextB);
        std::cout << "P1_TAIL_BASELINE quality_passed=" << baselineQuality << std::endl;
    }
    bool passed = true;
    Tensor hostLogits({2, static_cast<int64_t>(firstA.size())}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    {
        ContinuousBatchingProbe observer(runtime, stream);
        auto const samplerBefore = observer.samplingBytes();
        SequenceStepRuntime steps(runtime, stream);
        auto rejects = [&](auto operation, char const* label) {
            bool rejected = false;
            try
            {
                operation();
            }
            catch (std::logic_error const&)
            {
                rejected = true;
            }
            require(rejected && steps.healthy(), label);
            std::cout << "REJECT " << label << " passed=1" << std::endl;
        };
        rejects([&] { SequenceStepRuntime duplicate(runtime, stream); }, "second_session");
        LLMGenerationRequest request{};
        LLMGenerationResponse response;
        require(!runtime.handleRequest(request, response, stream), "Legacy call entered a leased runtime");
        auto finish = [&](Tensor const& logits) {
            steps.completeStep();
            size_t const elements = logits.getShape().volume();
            CUDA_CHECK(cudaMemcpyAsync(hostLogits.rawPointer(), logits.rawPointer(), elements * sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::vector<float>> values;
            for (int64_t row = 0; row < logits.getShape()[0]; ++row)
            {
                float const* begin = hostLogits.dataPointer<float>() + row * firstA.size();
                values.emplace_back(begin, begin + firstA.size());
            }
            return values;
        };
        SequenceOptions optionsA;
        optionsA.maxOutputTokens = 3;
        optionsA.temperature = 0.0F;
        SequenceOptions optionsB;
        optionsB.maxOutputTokens = 7;
        optionsB.seed = 123;
        auto a = steps.acquire(1, promptA, optionsA);
        auto b = steps.acquire(2, promptB, optionsB);
        auto const& logitsA = steps.beginPrefill(a, 65);
        require(steps.state(a).committedTokens() == 0, "Forward published state before completion");
        rejects([&] { steps.release(a); }, "release_in_flight");
        rejects([&] { steps.beginPrefill(b, 64); }, "overlapping_forward");
        passed = compare("native_prefill_A", firstA, finish(logitsA)[0], true) && passed;
        require(steps.state(a).output().empty(), "Prefill sampled unexpectedly");
        steps.acceptToken(a, greedy(firstA));
        require(steps.state(a).committedTokens() == 65, "Sample committed to KV early");
        observer.observeEndpoint(a.slot, 65);
        observer.snapshot(a.slot);
        finish(steps.beginPrefill(b, 64));
        observer.verifySnapshot();
        observer.observeEndpoint(b.slot, 64);
        observer.snapshot(b.slot);
        passed = compare("native_decode_A_while_B_prefills", nextA, finish(steps.beginDecode({a, {}}, 1))[0], true)
            && passed;
        observer.verifySnapshot();
        observer.observeEndpoint(a.slot, 66);
        observer.snapshot(a.slot);
        finish(steps.beginPrefill(b, 64));
        auto const tail = finish(steps.beginPrefill(b, 1))[0];
        passed = compare("native_sampling_free_tail_matches_P1", chunkFirstB, tail, true) && passed;
        require(steps.state(b).output().empty() && steps.state(b).committedTokens() == 129,
            "Prompt tail generated a token or committed incorrectly");
        observer.verifySnapshot();
        steps.acceptToken(b, greedy(firstB), 3);
        passed = compare("native_decode_B_after_tail_matches_P1", chunkNextB, finish(steps.beginDecode({b, {}}, 1))[0],
                     true)
            && passed;
        require(steps.state(b).randomCounter() == 3 && steps.state(a).randomCounter() == 0,
            "Random counters crossed requests");

        observer.observeEndpoint(b.slot, 130);
        observer.snapshot(b.slot);
        steps.finish(a);
        steps.release(a);
        auto c = steps.acquire(3, promptA, optionsA);
        require(c.slot == a.slot && c.generation != a.generation, "Slot was not safely reused");
        rejects([&] { steps.release(a); }, "stale_release");
        passed = compare("native_reused_slot_C", firstA, finish(steps.beginPrefill(c, 65))[0], true) && passed;
        observer.verifySnapshot();
        require(steps.state(b).options().maxOutputTokens == 7 && steps.state(b).output().size() == 1,
            "Partner request lost independent state");
        steps.release(c);
        steps.release(b);

        a = steps.acquire(4, promptA, optionsA);
        b = steps.acquire(5, promptB, optionsB);
        finish(steps.beginPrefill(a, 65));
        finish(steps.beginPrefill(b, 129));
        steps.acceptToken(a, greedy(firstA));
        steps.acceptToken(b, greedy(firstB));
        rejects([&] { steps.beginDecode({a, a}, 2); }, "duplicate_decode_slot");
        rejects([&] { steps.beginDecode({b, a}, 2); }, "unordered_decode_slots");
        auto const pair = finish(steps.beginDecode({a, b}, 2));
        passed = compare("native_unequal_batch_A", nextA, pair[0], true) && passed;
        passed = compare("native_unequal_batch_B", nextB, pair[1], true) && passed;
        require(steps.state(a).committedTokens() == 66 && steps.state(b).committedTokens() == 130,
            "Per-row endpoints did not commit independently");
        steps.release(a);
        steps.release(b);
        require(observer.samplingBytes() == samplerBefore, "Forward-only steps touched sampler output scratch");
        std::cout << "FORWARD_ONLY sampler_unchanged=1 passed=1" << std::endl;
    }

    LLMGenerationRequest legacy{};
    legacy.requests.resize(1);
    legacy.formattedRequests.resize(1);
    legacy.requests[0].messages.push_back({"user", {{"text", "Synthetic pretokenized regression fixture"}}});
    legacy.preTokenizedInputIds = {promptA};
    legacy.temperature = 0.0F;
    legacy.topK = 1;
    legacy.topP = 1.0F;
    legacy.maxGenerateLength = 2;
    legacy.applyChatTemplate = false;
    LLMGenerationResponse response;
    require(runtime.handleRequest(legacy, response, stream), "Legacy API failed after step session destruction");
    require(
        response.outputIds.size() == 1 && response.outputIds[0] == std::vector<int32_t>{greedy(firstA), greedy(nextA)},
        "Legacy output changed after releasing the step lease");
    std::cout << "LEGACY_RESTORED passed=1" << std::endl;
    std::cout << "P2_STATE_GATE passed=" << passed << " full_chunk_quality=separate_P1_TAIL_BASELINE" << std::endl;
    return passed;
}
} // namespace rt
} // namespace trt_edgellm

int main(int argc, char** argv)
{
    if (argc != 3 && (argc != 4 || std::string(argv[3]) != "--steps"))
    {
        std::cerr << "Usage: continuous_batching_probe ENGINE_DIR CHECKPOINT_DIR [--steps]\n";
        return 2;
    }
    cudaStream_t stream{};
    try
    {
        auto plugin = trt_edgellm::loadEdgellmPluginLib();
        trt_edgellm::rt::require(plugin != nullptr, "Plugin loading failed");
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bool passed = false;
        {
            trt_edgellm::tokenizer::Tokenizer tokenizer;
            trt_edgellm::rt::require(tokenizer.loadFromHF(argv[1], false), "Tokenizer loading failed");
            trt_edgellm::rt::LLMRankRuntime runtime(argv[1], "", {}, std::nullopt, stream,
                trt_edgellm::rt::ParallelMapping{}, tokenizer, trt_edgellm::rt::ContextCacheConfig{}, argv[2], "");
            passed = argc == 4 ? trt_edgellm::rt::runStepTests(runtime, tokenizer, stream)
                               : trt_edgellm::rt::runProbe(runtime, tokenizer, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::cout << "PROBE_RESULT passed=" << passed << std::endl;
        return passed ? 0 : 1;
    }
    catch (std::exception const& error)
    {
        std::cerr << "PROBE_ERROR " << error.what() << std::endl;
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
        return 1;
    }
}
