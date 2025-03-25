#pragma once

#include "runtime.cuh"
#include "compiler.cuh"

namespace deep_gemm::jit
{

class JITTuner
{
public:
    using BuildArgs = std::tuple<int,                  // shape_n
                                 int,                  // shape_k
                                 int,                  // block_m
                                 int,                  // block_n
                                 int,                  // block_k
                                 int,                  // num_groups
                                 int,                  // num_stages
                                 int,                  // num_tma_multicast
                                 deep_gemm::GemmType>; // gemm_type

    using RuntimeArgs = std::vector<void *>; // 存在参数数量和位置变化

    using SpaceType = std::vector<std::unordered_map<std::string, std::string>>;

public:

    JITTuner() = default;

    Runtime* compile_and_tune(const std::string&                           name,
                              std::unordered_map<std::string, std::string> keys,
                              const SpaceType&                             space,
                              const BuildArgs&                             build_args,
                              const RuntimeArgs&                           runtime_args)
    {
        // 对 keys 排序，确保一致性
        std::vector<std::pair<std::string, std::string>> sorted_keys(keys.begin(), keys.end());
        std::sort(sorted_keys.begin(), sorted_keys.end());

        std::string key_signature = name + "{";
        for (const auto& [k, v] : sorted_keys) {
            key_signature += k + ":" + v + ",";
        }
        key_signature += "}";

        if (tuned.find(key_signature) != tuned.end()) {
            if (std::getenv("DG_JIT_DEBUG")) {
                std::cout << "Using cached JIT kernel " << name << " with keys " << key_signature << std::endl;
            }
            return tuned[key_signature];
        }

        if (std::getenv("DG_JIT_DEBUG")) {
            std::cout << "Auto-tuning JIT kernel " << name << " with keys " << key_signature << std::endl;
        }

        // 保证最少执行一次
        if (space.empty()) {
            space.push_back({});
        }
        std::vector<std::pair<Runtime*, std::unordered_map<std::string, std::string>>> kernels;
        for (const auto& tuned_keys : space)
        {
            std::unordered_map<std::string, std::string> full_keys = keys;
            full_keys.insert(tuned_keys.begin(), tuned_keys.end());
            auto& runtime = getGlobalCompiler().build(build_args[0], // uint32_t const shape_n
                                                      build_args[1], // uint32_t const shape_k
                                                      build_args[2], // uint32_t const block_m
                                                      build_args[3], // uint32_t const block_n
                                                      build_args[4], // uint32_t const block_k
                                                      build_args[5], // uint32_t const num_groups
                                                      build_args[6], // uint32_t const num_stages
                                                      build_args[7], // uint32_t const num_tma_multicast
                                                      gemm_type);    // deep_gemm::GemmType const gemm_type
            kernels.emplace_back(runtime, tuned_keys);
        }

        Runtime* best_runtime;
        float best_time = std::numeric_limits<float>::max();
        std::unordered_map<std::string, std::string> best_keys;

        cudaEvent_t start_event, end_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        for (auto& [runtime, tuned_keys] : kernels) {
            float elapsed_time = 0.0f;
            if (space.size() > 1) {
                if (*runtime(runtime_args) != 0) continue;

                cudaEventRecord(start_event);
                for (int i = 0; i < 20; ++i) {
                    DG_HOST_ASSERT(*runtime(args) == 0);
                }
                cudaEventRecord(end_event);
                cudaEventSynchronize(end_event);
                cudaEventElapsedTime(&elapsed_time, start_event, end_event);
            }

            if (elapsed_time < best_time) {
                best_runtime = runtime;
                best_time = elapsed_time;
                best_keys = tuned_keys;
            }
        }
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);

        DG_HOST_ASSERT(best_runtime->is_valid());
        tuned[key_signature] = best_runtime;
        return best_runtime;
    }

    Runtime* get_best_runtime(const std::string&                           name,
                              std::unordered_map<std::string, std::string> keys)
    {
        std::vector<std::pair<std::string, std::string>> sorted_keys(keys.begin(), keys.end());
        std::sort(sorted_keys.begin(), sorted_keys.end());

        std::string key_signature = name + "{";
        for (const auto& [k, v] : sorted_keys) {
            key_signature += k + ":" + v + ",";
        }
        key_signature += "}";

        if (tuned.find(key_signature) != tuned.end()) {
            if (std::getenv("DG_JIT_DEBUG")) {
                std::cout << "Using cached JIT kernel " << name << " with keys " << key_signature << std::endl;
            }
            return tuned[key_signature];
        }

        return nullptr;
    }

private:

    std::unordered_map<std::string, Runtime> tuned;
};

}