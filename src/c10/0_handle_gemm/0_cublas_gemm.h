#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils.h"

NAMESPACE_C1000_BEGIN

template<typname T>
void run_cublas_gemm(cublasHandle_t handle,
                     int            M,
                     int            N,
                     int            K,
                     float          alpha,
                     float*         A,
                     float*         B,
                     float          beta,
                     float*         C)
{
    // cuBLAS uses column-major order. So we change the order of our row-major A & B
    // since (B^T * A^T)^T = (A * B)
    if constexpr (std::is_same_v<T, float>)
    {
        // this runs cuBLAS in full fp32 mode
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                    N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    else if constexpr (std::is_same_v<T, half>)
    {
        // This runs cuBLAS with mixed precision (performing the mul with operands
        // downcast to bf16), which is ~4x faster
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                    N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        // This runs cuBLAS with mixed precision (performing the mul with operands
        // downcast to bf16), which is ~4x faster
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                    N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    else
    {
        std::runtime_error("Unknown Data Type");
    }
}

NAMESPACE_C1000_END