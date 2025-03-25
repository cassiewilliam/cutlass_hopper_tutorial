#include <thrust/transform_reduce.h>
#include <cuda_bf16.h>

#include "utils.cuh"
#include "jit_utils.cuh"
#include "deep_gemm_runner.h"

float max_absolute_error(const thrust::host_vector<__nv_bfloat16>& ref,
                         const thrust::host_vector<__nv_bfloat16>& result)
{
    if (ref.size() != result.size())
    {
        std::cerr << "Error: Tensor size mismatch! (" << ref.size() << " vs " << result.size() << ")\n";
        return -1.0f;
    }

    float max_abs_diff = c108::EPSILON;
    for (int i = 0; i < ref.size(); ++i)
    {
        float abs_diff = fabs(static_cast<float>(result[i]) - static_cast<float>(ref[i]));

        max_abs_diff = (abs_diff > max_abs_diff) ? abs_diff : max_abs_diff; 
    }

    return max_abs_diff;
}

int main()
{
    int m = 256;
    int n = 15360;
    int k = 5120;

    cudaStream_t stream = 0;

    auto gemm_runner = std::make_shared<c108::DeepGemmRunner>();

    deep_gemm::GemmType gemm_type = deep_gemm::GemmType::Normal;
    gemm_runner->tunning(m, n, k, 1, gemm_type);

    thrust::host_vector<float> host_A(m * k);
    thrust::host_vector<float> host_B(n * k);
    thrust::host_vector<float> host_C(m * n);

    // Initialize the tensors
    c108::random_initialize(host_A.data(), m * k, -128.0f, 128.0f);
    c108::random_initialize(host_B.data(), n * k, -128.0f, 128.0f);
    for (int j = 0; j < m*n; ++j)
        host_C[j] = float(0);

    thrust::device_vector<float> device_A = host_A;
    thrust::device_vector<float> device_B = host_B;
    thrust::device_vector<float> device_C = host_C;

    cublasHandle_t handle;
    cublasCreate(&handle);
    c108::matmul_cublas(thrust::raw_pointer_cast(device_A.data()),
                        thrust::raw_pointer_cast(device_B.data()),
                        thrust::raw_pointer_cast(device_C.data()),
                        m, k, n, handle);
    cublasDestroy(handle);

    thrust::device_vector<__nv_fp8_e4m3> lhs_activation_a_x_fp8(m * k);
    thrust::device_vector<float> lhs_activation_a_x_scales(m * ceil_div(k, 128) * 128);
    c108::per_token_cast_to_fp8(thrust::raw_pointer_cast(device_A.data()),
                                thrust::raw_pointer_cast(lhs_activation_a_x_scales.data()),
                                thrust::raw_pointer_cast(lhs_activation_a_x_fp8.data()),
                                m,
                                k,
                                stream);

    thrust::device_vector<__nv_fp8_e4m3> rhs_weight_b_y_fp8(n * k);
    thrust::device_vector<float> rhs_weight_b_y_scales(ceil_div(n, 128) * 128 * ceil_div(k, 128) * 128);
    c108::per_block_cast_to_fp8(thrust::raw_pointer_cast(device_B.data()),
                                thrust::raw_pointer_cast(rhs_weight_b_y_scales.data()),
                                thrust::raw_pointer_cast(rhs_weight_b_y_fp8.data()),
                                n,
                                k,
                                stream);

    thrust::device_vector<__nv_bfloat16> result(m * n);

    gemm_runner->gemm(m, n, k,
                      (void *)(thrust::raw_pointer_cast(lhs_activation_a_x_fp8.data())),
                      (float *)(thrust::raw_pointer_cast(lhs_activation_a_x_scales.data())),
                      (void *)(thrust::raw_pointer_cast(rhs_weight_b_y_fp8.data())),
                      (float *)(thrust::raw_pointer_cast(rhs_weight_b_y_scales.data())),
                      (void *)(thrust::raw_pointer_cast(result.data())));

    thrust::host_vector<__nv_bfloat16> host_result_ref = device_C;
    thrust::host_vector<__nv_bfloat16> host_result = result;

    std::cout << max_absolute_error(host_result_ref, host_result);

    std::cout << "run test deep gemm success" << std::endl;

    return 0;
}

