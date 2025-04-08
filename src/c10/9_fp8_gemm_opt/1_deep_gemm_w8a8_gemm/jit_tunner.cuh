#pragma once

#include "runtime.cuh"
#include "compiler.cuh"
#include "utils.cuh"

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

    using SpaceType = std::vector<std::unordered_map<std::string, std::string>>;

public:

    JITTuner() = default;

    template <typename... Args>
    Runtime* compile_and_tune(const std::string&                           name,
                              std::unordered_map<std::string, std::string> keys,
                              SpaceType&                                   space,
                              const BuildArgs&                             build_args,
                              Args&&...                                    runtime_args)
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
            space.push_back(std::unordered_map<std::string, std::string>{});
        }
        std::vector<std::pair<Runtime*, std::unordered_map<std::string, std::string>>> kernels;
        for (const auto& tuned_keys : space)
        {
            std::unordered_map<std::string, std::string> full_keys = keys;
            full_keys.insert(tuned_keys.begin(), tuned_keys.end());
            auto&& runtime = getGlobalCompiler().build(std::get<0>(build_args),  // uint32_t const shape_n
                                                       std::get<1>(build_args),  // uint32_t const shape_k
                                                       std::get<2>(build_args),  // uint32_t const block_m
                                                       std::get<3>(build_args),  // uint32_t const block_n
                                                       std::get<4>(build_args),  // uint32_t const block_k
                                                       std::get<5>(build_args),  // uint32_t const num_groups
                                                       std::get<6>(build_args),  // uint32_t const num_stages
                                                       std::get<7>(build_args),  // uint32_t const num_tma_multicast
                                                       std::get<8>(build_args)); // deep_gemm::GemmType const gemm_type
            kernels.emplace_back(runtime, tuned_keys);
        }

        Runtime* best_runtime = nullptr;
        float best_time = std::numeric_limits<float>::max();
        std::unordered_map<std::string, std::string> best_keys;

        cudaEvent_t start_event, end_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        for (auto& [runtime, tuned_keys] : kernels) {
            float elapsed_time = 0.0f;
            if (space.size() > 1) {

                try {
                    (*runtime)(std::forward<Args>(runtime_args)...);
                } catch (...) {
                    continue;
                }

                cudaEventRecord(start_event);
                for (int i = 0; i < 20; ++i) {
                    (*runtime)(std::forward<Args>(runtime_args)...);
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

    std::unordered_map<std::string, Runtime*> tuned;
};

}