#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils.h"

NAMESPACE_C1000_BEGIN

template<const uint kBlockSize>
__global__ void sgemm_global_mem_coalesce(int          M,
                                          int          N,
                                          int          K,
                                          float        alpha,
                                          const float* A,
                                          const float* B,
                                          float        beta,
                                          float*       C)
{
    const int x = blockIdx.x * kBlockSize + threadIdx.x / kBlockSize;
    const int y = blockIdx.y * kBlockSize + threadIdx.x % kBlockSize;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N)
    {
        float temp = 0.0;
        for (int i = 0; i < K; ++i)
        {
            // A矩阵是每个Warp获取一行，B矩阵是一个Warp获取一行中的32个数据，
            // 然后按照K方向进行向下访问，B矩阵可以合并访问，A矩阵可以一个线程访问其它线程广播，极大的改善了访存效率
            temp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}

void run_sgemm_coalesce(int    M,
                        int    N,
                        int    K,
                        float  alpha,
                        float* A,
                        float* B,
                        float  beta,
                        float* C)
{
    dim3 grid_dim(UP_DIV(M, 32), UP_DIV(N, 32));
    dim3 block_dim(32 * 32);

    sgemm_global_mem_coalesce<32><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}

NAMESPACE_C1000_END