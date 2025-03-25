/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

#include <exception>

#ifdef __CLION_IDE__
__host__ __device__ __forceinline__ void host_device_printf(char const* format, ...)
{
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

class AssertionException : public std::exception
{
private:
    std::string message{};

public:
    explicit AssertionException(std::string const& message)
        : message(message)
    {
    }

    char const* what() const noexcept override
    {
        return message.c_str();
    }
};

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (not(cond))                                                                                                 \
        {                                                                                                              \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);                             \
            throw AssertionException("Assertion failed: " #cond);                                                      \
        }                                                                                                              \
    } while (0)
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (not(cond))                                                                                                 \
        {                                                                                                              \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);                             \
            asm("trap;");                                                                                              \
        }                                                                                                              \
    } while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

template <typename T>
__device__ __host__ constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}


template <typename Callable, size_t... Indices>
auto unpack_and_invoke(Callable&& runtime, std::vector<void*>& runtime_args, std::index_sequence<Indices...>)
{
    return std::forward<Callable>(runtime)(
        *reinterpret_cast<void**>(runtime_args[0]),         // void*        mat_a
        *reinterpret_cast<int*>(runtime_args[1]),           // int          lda
        *reinterpret_cast<void**>(runtime_args[2]),         // void*        mat_b
        *reinterpret_cast<int*>(runtime_args[3]),           // int          ld_b
        *reinterpret_cast<void**>(runtime_args[4]),         // void*        mat_d
        *reinterpret_cast<int*>(runtime_args[5]),           // int          ld_d
        reinterpret_cast<float*>(runtime_args[6]),          // float*       scales_a
        reinterpret_cast<float*>(runtime_args[7]),          // float*       scales_b
        *reinterpret_cast<uint32_t*>(runtime_args[8]),      // uint32_t     shape_m,
        reinterpret_cast<int*>(runtime_args[9]),            // int*         grouped_layout
        *reinterpret_cast<cudaStream_t*>(runtime_args[10]), // cudaStream_t stream
        *reinterpret_cast<int*>(runtime_args[11]),          // int          num_sms
        *reinterpret_cast<uint32_t*>(runtime_args[12])      // uint32_t     smem_size
    );
}

template <typename Callable>
auto invoke_runtime(Callable&& runtime, std::vector<void*>& runtime_args)
{
    if (runtime_args.size() != 13)
    {
        throw std::runtime_error("Argument size mismatch! Expected 13 arguments.");
    }

    return unpack_and_invoke(std::forward<Callable>(runtime), runtime_args, std::make_index_sequence<13>{});
}

namespace c108
{

// NOTE(Alan): fp8 min max match with torch
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr float EPSILON = 1e-4f;

// CUDA核函数：计算amax数值
__global__ void compute_amax_per_token(const float* tensor, float* amax, int m, int n)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= m) return;

    float max_val = 0.0f;

    // 每个线程遍历自己的部分列，取最大值
    for (int i = tid; i < n; i += blockDim.x)
    {
        max_val = fmaxf(max_val, fabsf(tensor[row * n + i]));
    }

    // 使用 shfl 在warp内部进行规约
    for (int offset = 16; offset > 0; offset /= 2)
    {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // 仅 0 号线程写入结果
    if (tid == 0)
    {
        amax[row] = fmaxf(max_val, EPSILON);
    }
}

__global__ void cast_to_fp8_with_scale(const float* tensor, float* amax, __nv_fp8_e4m3* fp8_tensor, int m, int n)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < m && col < n)
    {
        float scale = FP8_E4M3_MAX / amax[row];
        float value = fmax(-FP8_E4M3_MAX, fmin(tensor[row * n + col] * scale, FP8_E4M3_MAX));

        fp8_tensor[row * n + col] = static_cast<__nv_fp8_e4m3>(value);
    }
}

// 将Tensor进行Reshape，按照Block128x128的形式进行排列
__global__ void reshape_tensor_to_block128x128(const float* tensor, float* reshaped_tensor, int m, int n)
{
    int row = blockIdx.x * 128 + threadIdx.x;
    int col = blockIdx.y * 128 + threadIdx.y;

    if (row < m && col < n)
    {
        int src_idx = row * n + col;
        int dst_idx = row * ceil_div(n, 128) * 128 + col;

        reshaped_tensor[dst_idx] = tensor[src_idx];
    }
}

// 将Tensor进行Reshape，将Block128x128的形式转化为m * n的形式
__global__ void reshape_block128x128_to_tensor(const __nv_fp8_e4m3* tensor, __nv_fp8_e4m3* reshaped_fp8_tensor, int m, int n)
{
    int row = blockIdx.x * 128 + threadIdx.x;
    int col = blockIdx.y * 128 + threadIdx.y;

    if (row < m && col < n)
    {
        int src_idx = row * ceil_div(n, 128) * 128 + col;
        int dst_idx = row * n + col;

        reshaped_fp8_tensor[dst_idx] = tensor[src_idx];
    }
}

static void random_initialize(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

// cuBLAS 矩阵乘法 C = A * B^T
static void matmul_cublas(const float* d_A,
                          const float* d_B,
                          float* d_C,
                          int m,
                          int k,
                          int n,
                          cublasHandle_t handle)
{
    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_T, // B 需要转置
                CUBLAS_OP_N, // A 保持不变
                n,           // 结果矩阵 C 的列数 (B^T 的行数)
                m,           // 结果矩阵 C 的行数 (A 的行数)
                k,           // A 的列数 (B 的列数)
                &alpha, 
                d_B, k,      // B (k × n)，按列主序存储
                d_A, k,      // A (m × k)，按列主序存储
                &beta, 
                d_C, n);     // C (m × n)，按列主序存储
}


// 此处的token不是真实的token，可以认为把原始的input[B * S, H] 按照128的H维度切分，进行reshape
// 那么input最终shape变为了[B * S * (H / 128), 128]，那么此时m就为B * S * (H / 128)， n为128
static void per_token_cast_to_fp8(float* tensor, float* amax, __nv_fp8_e4m3* fp8_tensor, int m, int n, cudaStream_t stream)
{
    DG_HOST_ASSERT(n % 128 == 0);

    int reshape_m = m * n / 128;
    int reshape_n = 128;

    // 计算当前token的amax
    compute_amax_per_token<<<reshape_m, reshape_n, 0, stream>>>(tensor, amax, reshape_m, reshape_n);
    cudaDeviceSynchronize();

    // 对输入tensor进行量化
    cast_to_fp8_with_scale<<<reshape_m, reshape_n, 0, stream>>>(tensor, amax, fp8_tensor, reshape_m, reshape_n);
    cudaDeviceSynchronize();
}

static void per_block_cast_to_fp8(float* tensor, float* amax, __nv_fp8_e4m3* fp8_tensor, int m, int n, cudaStream_t stream)
{
    DG_HOST_ASSERT(n % 128 == 0);
    
    int padded_m = ceil_div<int>(m, 128) * 128;
    int padded_n = ceil_div<int>(n, 128) * 128;

    // Step1. Padding Tensor
    float* padded_tensor;
    cudaMalloc(&padded_tensor, padded_m * padded_n * sizeof(float));
    cudaMemset(padded_tensor, 0, padded_m * padded_n * sizeof(float));
    cudaMemcpy2D(padded_tensor, 
                 padded_n * sizeof(float),
                 tensor,
                 n * sizeof(float),
                 n * sizeof(float),
                 m,
                 cudaMemcpyDeviceToDevice);

    int reshape_m = padded_m / 128 * padded_n / 128;
    int reshape_n = 128 * 128;

    // Step2. Reshape Tensor
    float* reshaped_tensor;
    cudaMalloc(&reshaped_tensor, reshape_m * reshape_n * sizeof(float));
    cudaMemset(reshaped_tensor, 0, reshape_m * reshape_n * sizeof(float));
    dim3 block_size(128, 128);
    dim3 grid_size(ceil_div<int>(padded_m, 128), ceil_div<int>(padded_n, 128));
    reshape_tensor_to_block128x128<<<grid_size, block_size, 0, stream>>>(padded_tensor, reshaped_tensor, padded_m, padded_n);

    // Step3. Quantization Tensor
    // 计算当前block的amax
    compute_amax_per_token<<<reshape_m, reshape_n, 0, stream>>>(reshaped_tensor, amax, reshape_m, reshape_n);
    cudaDeviceSynchronize();

    __nv_fp8_e4m3* quantized_fp8_tensor;
    cudaMalloc(&quantized_fp8_tensor, reshape_m * reshape_n * sizeof(__nv_fp8_e4m3));
    cudaMemset(quantized_fp8_tensor, 0, reshape_m * reshape_n * sizeof(__nv_fp8_e4m3));
    // 对输入tensor进行量化
    cast_to_fp8_with_scale<<<reshape_m, reshape_n, 0, stream>>>(reshaped_tensor, amax, quantized_fp8_tensor, reshape_m, reshape_n);
    cudaDeviceSynchronize();

    // Step4. Inv Reshape Tensor
    __nv_fp8_e4m3* reshaped_fp8_tensor;
    cudaMalloc(&reshaped_fp8_tensor, padded_m * padded_n * sizeof(__nv_fp8_e4m3));
    cudaMemset(reshaped_fp8_tensor, 0, padded_m * padded_n * sizeof(__nv_fp8_e4m3));
    reshape_block128x128_to_tensor<<<grid_size, block_size, 0, stream>>>(quantized_fp8_tensor, reshaped_fp8_tensor, padded_m, padded_n);

    // Step4. Inv Padding Tensor
    cudaMemcpy2D(fp8_tensor, 
                 n * sizeof(float),
                 reshaped_fp8_tensor,
                 padded_n * sizeof(float),
                 n * sizeof(float),
                 m,
                 cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    cudaFree(padded_tensor);
    cudaFree(reshaped_tensor);
    cudaFree(quantized_fp8_tensor);
    cudaFree(reshaped_fp8_tensor);
}

}