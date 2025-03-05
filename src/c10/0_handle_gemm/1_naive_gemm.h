#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils.h"

NAMESPACE_C1000_BEGIN

/**
 * Matrix sizes:
 * M x K * K x N = M x N
 */
__global__ void sgemm_naive(int          M,
                            int          N,
                            int          K,
                            float        alpha,
                            const float* A,
                            const float* B,
                            float        beta,
                            float*       C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N)
    {
        float temp = 0.0f;

        for (int i = 0; i < K; ++i)
        {
            temp += A[x * K + i] * B[i * N + y];
        }

        // C = alpha * (A @ B) + beta * C
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}

void run_sgemm_naive(int M,
                     int N,
                     int K,
                     float  alpha,
                     float* A,
                     float* B,
                     float  beta,
                     float* C)
{
    dim3 grid_dim(UP_DIV(M, 32), UP_DIV(N, 32));
    dim3 block_dim(32, 32);

    sgemm_naive<<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}

NAMESPACE_C1000_END