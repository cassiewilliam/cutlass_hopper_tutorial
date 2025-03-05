#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils.h"

NAMESPACE_C1000_BEGIN

template<const int kBlockM,
         const int kBlockN,
         const int kBlockK,
         const int kThreadM>
__global__ void sgemm_1d_blocktiling(int          M,
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

    // 按照块大小kBlockM * kBlockK进行A Block的Shared Memory分配
    __shared__ float smem_A[kBlockM * kBlockK];
    // 按照块大小kBlockK * kBlockN进行B Block的Shared Memory分配
    __shared__ float smem_B[kBlockK * kBlockN];

    // 初始化Block块对应的坐标位置
    /// A是按行访问，所以需要乘以K，可以认为是Block Stride，K是在行方向进行循环
    A = A + block_x * kBlockM * K;
    /// B是按列访问，K是在列方向进行访问
    B = B + block_y * kBlockN;
    /// C是每次访问一个固定的块[kBlockSize * kBlockSize]
    C = C + block_x * kBlockM * N + block_y * kBlockN;

    // 计算一个(block_x, block_y)对应位置的块结果
    // 当前只考虑A和B分块大小一致并且和抛出的线程一样的情况
    // 如果不一致需要有不同的对应关系，使得问题变复杂
    assert(kBlockM * kBlockK == blockDim.x);
    assert(kBlockN * kBlockK == blockDim.x);
    {
        // thread block是按照 kBlockSize * kBlockSize 个线程进行分配的
        // 也就是获取二维block内部的(x, y)坐标索引, 对应到Block获取对应的数据
        const uint C_thread_y = threadIdx.x / kBlockN;
        const uint C_thread_x = threadIdx.x % kBlockN;

        const uint A_thread_y = threadIdx.x / kBlockK; // warp-level GEMM coalescing
        const uint A_thread_x = threadIdx.x % kBlockK;

        const uint B_thread_y = threadIdx.x / kBlockN; // warp-level GEMM coalescing
        const uint B_thread_x = threadIdx.x % kBlockN;

        // 分配thread-local的寄存器地址用来存储结果
        float reg_block_D[kThreadM] = {0.0};

        for (int k_idx = 0; k_idx < K; k_idx += kBlockK)
        {
            // A矩阵是每个Warp获取一行，B矩阵是一个Warp获取一行中的kBlockK个数据，
            // 然后按照K方向进行向下访问，B矩阵可以合并访问，A矩阵可以一个线程访问其它线程广播，极大的改善了访存效率
            smem_A[A_thread_y * kBlockK + A_thread_x] = A[A_thread_y * K + A_thread_x];
            smem_B[B_thread_y * kBlockN + B_thread_x] = B[B_thread_y * N + B_thread_x];

            // 等待一个[kBlockM * kBlockK]或者[kBlockK * kBlockN]的数据搬运完成，只会对一个Block内部的线程要求同步
            __syncthreads();

            // K for 循环自增
            A = A + kBlockK;
            B = B + kBlockK * N;

            // 计算每个线程的结果
            for (int block_k_idx = 0; block_k_idx < kBlockK; ++block_k_idx)
            {
                float elem_B = smem_B[block_k_idx * kBlockN + B_thread_x];

                // 每个线程需要计算kThreadM个元素的结果，这样可以复用smem_B中的数据读取，从而提升访存效率，
                for (int idx = 0; idx < kThreadM; ++idx)
                {
                    // 对于smem_A中的数据，按照Block角度进行向下平移，也可以很好的利用Cache
                    reg_block_D[idx] += smem_A[(A_thread_y + idx) * kBlockK + block_k_idx] * elem_B;
                }
            }

            // 防止部分线程过快导致Cache 被冲掉
            __syncthreads();
        }

        for (int idx = 0; idx < kThreadM; ++idx)
        {
            const int C_index = (C_thread_y * kThreadM + idx) * kBlockN + C_thread_x;
            C[C_index] = alpha * reg_block_D[idx] + beta * C[C_index];
        }
    }
}

void run_sgemm_1d_blocktiling(int    M,
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

    sgemm_1d_blocktiling<32><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}

NAMESPACE_C1000_END