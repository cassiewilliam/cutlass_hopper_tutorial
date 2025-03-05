#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils.h"

NAMESPACE_C1000_BEGIN

template<const uint kBlockSize>
__global__ void sgemm_shared_mem_block(int          M,
                                       int          N,
                                       int          K,
                                       float        alpha,
                                       const float* A,
                                       const float* B,
                                       float        beta,
                                       float*       C)
{
    // 整个矩阵按照kBlockSize * kBlockSize进行切块，(x, y)的坐标表示对应块的坐标
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;

    // 按照块大小kBlockSize * kBlockSize进行Shared Memory分配
    __shared__ float smem_A[kBlockSize * kBlockSize];
    __shared__ float smem_B[kBlockSize * kBlockSize];

    // 初始化Block块对应的坐标位置
    /// A是按行访问，所以需要乘以K，可以认为是Block Stride，K是在行方向进行循环
    A = A + block_x * kBlockSize * K;
    /// B是按列访问，K是在列方向进行访问
    B = B + block_y * kBlockSize;
    /// C是每次访问一个固定的块[kBlockSize * kBlockSize]
    C = C + block_x * kBlockSize * N + block_y * kBlockSize;

    // 计算一个(block_x, block_y)对应位置的块结果
    {
        // thread block是按照 kBlockSize * kBlockSize 个线程进行分配的
        // 也就是获取二维block内部的(x, y)坐标索引
        const uint thread_y = threadIdx.x / kBlockSize;
        const uint thread_x = threadIdx.x % kBlockSize;

        float temp = 0.0;
        for (int k_idx = 0; k_idx < K; k_idx += kBlockSize)
        {
            // A矩阵是每个Warp获取一行，B矩阵是一个Warp获取一行中的kBlockSize个数据，
            // 然后按照K方向进行向下访问，B矩阵可以合并访问，A矩阵可以一个线程访问其它线程广播，极大的改善了访存效率
            smem_A[thread_y * kBlockSize + thread_x] = A[thread_y * K + thread_x];
            smem_B[thread_y * kBlockSize + thread_x] = B[thread_y * N + thread_x];

            // 等待一个[kBlockSize * kBlockSize]的数据搬运完成，只会对一个Block内部的线程要求同步
            __syncthreads();

            // K for 循环自增
            A = A + kBlockSize;
            B = B + kBlockSize * N;

            
            for (int block_k_idx = 0; block_k_idx < kBlockSize; ++block_k_idx)
            {
                temp += smem_A[thread_y * kBlockSize + block_k_idx] * smem_B[block_k_idx * kBlockSize + thread_x];
            }

            // 防止部分线程过快导致Cache 被冲掉
            __syncthreads();
        }

        C[thread_y * N + thread_x] = alpha * temp + beta * C[thread_y * N + thread_x];
    }
}

void run_sgemm_smem_block(int    M,
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

    sgemm_shared_mem_block<32><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}

NAMESPACE_C1000_END