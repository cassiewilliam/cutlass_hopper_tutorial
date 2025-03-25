#include "utils.cuh"
#include "jit_utils.cuh"
#include "deep_gemm.h"

namespace c108
{

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
        matmul_cublas(device_A.data(),
                      device_B.data(),
                      device_C.data(),
                      m, k, n, handle);
        cublasDestroy(handle);

        thrust::device_vector<__nv_fp8_e4m3> lhs_activation_a_x_fp8(m * k);
        thrust::device_vector<float> lhs_activation_a_x_scales(m * ceil_div(k, 128) * 128);
        per_token_cast_to_fp8(device_A.data(),
                              lhs_activation_a_x_scales.data(),
                              lhs_activation_a_x_fp8.data(),
                              m,
                              k,
                              stream);

        thrust::device_vector<__nv_fp8_e4m3> rhs_weight_b_y_fp8(n * k);
        thrust::device_vector<float> rhs_weight_b_y_scales(ceil_div(n, 128) * 128 * ceil_div(k, 128) * 128);
        per_block_cast_to_fp8(device_B.data(),
                              rhs_weight_b_y_scales.data(),
                              rhs_weight_b_y_fp8.data(),
                              n,
                              k,
                              stream);

        thrust::device_vector<__nv_bfloat16> output(m * n);
        deep_gemm::jit::JITTuner::RuntimeArgs runtime_args{
            (void *)(lhs_activation_a_x_fp8.data()),
            (void *)(lhs_activation_a_x_scales.data()),
            (void *)(rhs_weight_b_y_fp8.data()),
            (void *)(rhs_weight_b_y_scales.data()),
            (void *)(output.data()),
            (void *)(&m),
            (void *)(&stream),
            (void *)(&std::get<0>(config)),
            (void *)(&std::get<5>(config))
        };

        m_jittuner->compile_and_tune(name, keys, space, build_args, runtime_args);
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
                          void*        A_scales,
                          void*        B,
                          void*        B_scales,
                          void*        C,
                          cudaStream_t stream = 0)
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

    deep_gemm::jit::JITTuner::RuntimeArgs runtime_args{
        A
        A_scales,
        B
        B_scales,
        C,
        (void *)(&m),
        (void *)(&stream),
        (void *)(&std::get<0>(config)), // num_sms
        (void *)(&std::get<5>(config))  // smem_size
    };
    (*runtime)(runtime_args);
}

}