#pragma once

#include <cuda_runtime.h>

#include "compiler.cuh"
#include "jit_tunner.cuh"

namespace c108
{

class DeepGemmRunner
{
public:

    DeepGemmRunner()
    {
        m_jittuner = std::make_shared<deep_gemm::jit::JITTuner>();
    }

    ~DeepGemmRunner() {}

    void tunning(int m, int n, int k, int num_groups, deep_gemm::GemmType type, cudaStream_t stream = 0);

    void gemm(int          m,
              int          n,
              int          k,
              void*        A,
              float*       A_scales,
              void*        B,
              float*       B_scales,
              void*        C,
              cudaStream_t stream = 0);

    void group_gemm_contiguous(int          m,
                                int          n,
                                int          k,
                                float        alpha,
                                void const*  A,
                                int          lda,
                                void const*  B,
                                int          ldb,
                                float        beta,
                                void*        C,
                                int          ldc,
                                cudaStream_t stream = 0);

    void group_gemm_masked(int          m,
                            int          n,
                            int          k,
                            float        alpha,
                            void const*  A,
                            int          lda,
                            void const*  B,
                            int          ldb,
                            float        beta,
                            void*        C,
                            int          ldc,
                            cudaStream_t stream = 0);

private:

    std::shared_ptr<deep_gemm::jit::JITTuner> m_jittuner;
};


void DeepGemmRunner::tunning(int m, int n, int k, int num_groups, deep_gemm::GemmType type, cudaStream_t stream)
{
    int num_sms = deep_gemm::jit::get_num_sms();

    // 将BLOCK_K设置为常数128，需要确认是否可以进行tunning
    int kBlockK = 128;

    bool is_grouped_contiguous = (type == deep_gemm::GemmType::GroupedContiguous);
    // return arguments
    // num_minimal_sms   : 在保证waves不变的情况下，最少需要多少个sm
    // best_block_m      : 当前配置下最优的block_m
    // best_block_n      : 当前配置下最优的block_n
    // num_stages        : 当前资源下最优的stages
    // num_tma_multicast : 是否使用TMA_MultiCast功能？
    // smem_size         : 当前配置下总共占用多少shared memory
    auto config = deep_gemm::jit::get_best_gemm_config(m, n, k, num_groups, num_sms, is_grouped_contiguous);

    // construct gemm args
    if (type == deep_gemm::GemmType::Normal)
    {
        std::string name = "gemm_fp8_fp16_bf16_nt";

        std::unordered_map<std::string, std::string> keys = {
            {"N",                 std::to_string(n)},
            {"K",                 std::to_string(k)},
            {"BLOCK_M",           std::to_string(std::get<1>(config))},
            {"BLOCK_N",           std::to_string(std::get<2>(config))},
            {"NUM_STAGES",        std::to_string(std::get<3>(config))},
            {"NUM_TMA_MULTICAST", std::to_string(std::get<4>(config))}
        };
        std::vector<std::unordered_map<std::string, std::string>> space = {};

        deep_gemm::jit::JITTuner::BuildArgs build_args{
            n,
            k,
            std::get<1>(config), // kBlockM
            std::get<2>(config), // kBlockN
            kBlockK,
            num_groups,
            std::get<3>(config), // kNumStages
            std::get<4>(config), // kNumTmaMulticast
            type
        };
        thrust::host_vector<float> host_A(m * k);
        thrust::host_vector<float> host_B(n * k);
        thrust::host_vector<float> host_C(m * n);

        // Initialize the tensors
        random_initialize(host_A.data(), m * k, -128.0f, 128.0f);
        random_initialize(host_B.data(), n * k, -128.0f, 128.0f);
        for (int j = 0; j < m*n; ++j)
            host_C[j] = float(0);

        thrust::device_vector<float> device_A = host_A;
        thrust::device_vector<float> device_B = host_B;
        thrust::device_vector<float> device_C = host_C;

        cublasHandle_t handle;
        cublasCreate(&handle);
        matmul_cublas(thrust::raw_pointer_cast(device_A.data()),
                      thrust::raw_pointer_cast(device_B.data()),
                      thrust::raw_pointer_cast(device_C.data()),
                      m, k, n, handle);
        cublasDestroy(handle);

        thrust::device_vector<__nv_fp8_e4m3> lhs_activation_a_x_fp8(m * k);
        thrust::device_vector<float> lhs_activation_a_x_scales(m * ceil_div(k, 128) * 128);
        per_token_cast_to_fp8(thrust::raw_pointer_cast(device_A.data()),
                              thrust::raw_pointer_cast(lhs_activation_a_x_scales.data()),
                              thrust::raw_pointer_cast(lhs_activation_a_x_fp8.data()),
                              m,
                              k,
                              stream);

        thrust::device_vector<__nv_fp8_e4m3> rhs_weight_b_y_fp8(n * k);
        thrust::device_vector<float> rhs_weight_b_y_scales(ceil_div(n, 128) * 128 * ceil_div(k, 128) * 128);
        per_block_cast_to_fp8(thrust::raw_pointer_cast(device_B.data()),
                              thrust::raw_pointer_cast(rhs_weight_b_y_scales.data()),
                              thrust::raw_pointer_cast(rhs_weight_b_y_fp8.data()),
                              n,
                              k,
                              stream);

        thrust::device_vector<__nv_bfloat16> output(m * n);

        uint32_t num_sms = std::get<0>(config);
        uint32_t smem_size = std::get<5>(config);
        m_jittuner->compile_and_tune(name, keys, space, build_args,
            (void *)(thrust::raw_pointer_cast(lhs_activation_a_x_fp8.data())), k,
            (void *)(thrust::raw_pointer_cast(rhs_weight_b_y_fp8.data())), k,
            (void *)(thrust::raw_pointer_cast(output.data())), n,
            (float *)(thrust::raw_pointer_cast(lhs_activation_a_x_scales.data())),
            (float *)(thrust::raw_pointer_cast(rhs_weight_b_y_scales.data())),
            m, nullptr, stream, num_sms, smem_size
        );
    }
    else if (type == deep_gemm::GemmType::GroupedContiguous)
    {
        std::runtime_error("Not Support Gemm Type");
    }
    else if (type == deep_gemm::GemmType::GroupedMasked)
    {
        std::runtime_error("Not Support Gemm Type");
    }
    else
    {
        std::runtime_error("Not Support Gemm Type");
    }
}

void DeepGemmRunner::gemm(int          m,
                          int          n,
                          int          k,
                          void*        A,
                          float*       A_scales,
                          void*        B,
                          float*       B_scales,
                          void*        C,
                          cudaStream_t stream)
{

    std::string name = "gemm_fp8_fp16_bf16_nt";


    int num_sms = deep_gemm::jit::get_num_sms();

    // return arguments
    // num_minimal_sms   : 在保证waves不变的情况下，最少需要多少个sm
    // best_block_m      : 当前配置下最优的block_m
    // best_block_n      : 当前配置下最优的block_n
    // num_stages        : 当前资源下最优的stages
    // num_tma_multicast : 是否使用TMA_MultiCast功能？
    // smem_size         : 当前配置下总共占用多少shared memory
    auto config = deep_gemm::jit::get_best_gemm_config(m, n, k, 1, num_sms);

    std::unordered_map<std::string, std::string> keys = {
        {"N",                 std::to_string(n)},
        {"K",                 std::to_string(k)},
        {"BLOCK_M",           std::to_string(std::get<1>(config))},
        {"BLOCK_N",           std::to_string(std::get<2>(config))},
        {"NUM_STAGES",        std::to_string(std::get<3>(config))},
        {"NUM_TMA_MULTICAST", std::to_string(std::get<4>(config))}
    };

    auto runtime = m_jittuner->get_best_runtime(name, keys);

    uint32_t num_minimal_sms = std::get<0>(config);
    uint32_t smem_size = std::get<5>(config);

    (*runtime)(A, k, B, k, C, n, A_scales, B_scales, m, nullptr, stream, num_minimal_sms, smem_size);
}


}