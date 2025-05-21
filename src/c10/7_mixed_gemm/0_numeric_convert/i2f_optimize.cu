#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "include/gemm_util.h"
#include <cuda_runtime.h>
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <bitset>

#include <cuda_fp16.h>

__device__ uint4 dequantize_s4_to_fp16x2_cvt(uint32_t const& source)
{
    uint4 result;  // 用于存储转换后的 8 个 fp16 值（每个 uint4 存储 4 个值）

    // 提取 8 个 int4 数据并转换为 fp16
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // 提取 4 位数据
        int int4_value = (source >> (i * 4)) & 0xF;

        // NOTE: int4 2 float16打开这个
        // 转换为有符号值 (-8 到 7)
        // if (int4_value >= 8) {
        //     int4_value -= 16;
        // }

        // 将 int4 转换为 float16
        __half fp16_value = __float2half((float)int4_value);
        ((uint16_t*)&result)[i] = *reinterpret_cast<uint16_t*>(&fp16_value);
    }

    return result;
}

// NOTE: 期望输入的数据排布是
//       input index ｜0｜1｜2｜3｜4｜5｜6｜7｜
//       elemnt      ｜0｜2｜4｜6｜1｜3｜5｜7｜
//       得到输出的数据排布是
//       output index｜0｜1｜2｜3｜4｜5｜6｜7｜
//       elemnt      ｜0｜1｜2｜3｜4｜5｜6｜7｜
//       如果输入数据没进行预先排列，那么等到的数据排布是
//       output index｜0｜1｜2｜3｜4｜5｜6｜7｜
//       elemnt      ｜0｜4｜1｜5｜2｜6｜3｜7｜
//
//      当前kernel正是利用了预先重排数据的格式特性，并且结合PTX指令进行了细致优化
//      从而取得了显著的成果
__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
{
    uint4 result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    // BOTTOM_MASK: 0000 0000 0000 1111 0000 0000 0000 1111
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    // TOP_MASK: 0000 0000 1111 0000 0000 0000 1111 0000
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    // i4s -> fp16前缀: 0110 0100 0000 0000 0110 0100 0000 0000
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;
    // NOTE: 0x6400 0, 11010, 0000000000
    //              +, 25-15, 1.0 = 1024

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    // elt_67 to fp16 without having to shift them to the bottom bits before hand.

    // h = [e0,e2,e4,e6,e1,e3,e5,e7]，数据表示
    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    // h = [0,0,0,e6,0,0,0,e7]，mask数据表示
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[0])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    // h = [0,0,e5,0,0,0,e6,0]，mask数据表示，左移动了4位，需要除以16，并且符号为变为了第8位，所以需要减去2^8=64
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[1])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[2])
                    : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[3])
                    : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
    // half2 ctor. In this case, I chose performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // NOTE: int4 2 float16打开这个
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    // NOTE(Alan): 对于最小单位8bit的数据，低4bit可以直接操作，高4bit需要移4位，也就是除以16
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00; // NOTE: 0x0010,1100,0000,000 = 1.0 * 2^-4 = 1/16
    // This is the half2 {-72, -72} represented as an integer.
    // NOTE: int4 2 float16打开这个
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400; // NOTE: 符号位

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;
}

__global__ void test_dequantize(uint32_t* input, uint4* output, int n, bool opt = false)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (opt)
            output[idx] = dequantize_s4_to_fp16x2(input[idx]);
        else
            output[idx] = dequantize_s4_to_fp16x2_cvt(input[idx]);
    }
}

int main(int argc, char const** argv)
{
    constexpr int num_elems = 1024 * 1024 * 128; // 4 * 128M elements
    uint32_t* h_data = (uint32_t*)malloc(num_elems * sizeof(int));
    uint32_t* d_data;
    uint4* cvt_result;
    uint4* opt_result;

    // Initialize host data
    for (int i = 0; i < num_elems; ++i) {
        h_data[i] = rand();
    }

    bool opt = false;
    int iterations = 100;
    cutlass::CommandLine cmd(argc, argv);

    if (cmd.check_cmd_line_flag("opt"))
    {
        opt = true;
    }

    cmd.get_cmd_line_argument("iterations", iterations);

    cudaMalloc(&d_data, num_elems * sizeof(uint32_t));
    cudaMemcpy(d_data, h_data, num_elems * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc(&cvt_result, num_elems * sizeof(uint4));
    cudaMalloc(&opt_result, num_elems * sizeof(uint4));

    constexpr int grid_x = 32;
    constexpr int grid_y = num_elems / 128 / 32;
    uint2 dispatch_grid_dim = {grid_x, grid_y};
    uint2 cta_dim = {16, 8};
    uint max_tile_width = 16;

    dim3 threads_per_block(cta_dim.x, cta_dim.y);
    dim3 num_blocks((dispatch_grid_dim.x + cta_dim.x - 1) / cta_dim.x, (dispatch_grid_dim.y + cta_dim.y - 1) / cta_dim.y);

    // Compare Results
    {
        test_dequantize<<<num_blocks, threads_per_block>>>(d_data, cvt_result, num_elems, false);
        test_dequantize<<<num_blocks, threads_per_block>>>(d_data, opt_result, num_elems, true);

        cudaDeviceSynchronize();

        // Copy data back to host and validate
        uint4* h_cvt_result = (uint4*)malloc(num_elems * sizeof(uint4));
        cudaMemcpy(h_cvt_result, cvt_result, num_elems * sizeof(uint4), cudaMemcpyDeviceToHost);

        uint4* h_opt_result = (uint4*)malloc(num_elems * sizeof(uint4));
        cudaMemcpy(h_opt_result, opt_result, num_elems * sizeof(uint4), cudaMemcpyDeviceToHost);

        std::cout << "Binary: " << std::bitset<32>(*h_data) << std::endl;

        int index_map[8] = {0, 4, 1, 5, 2, 6, 3, 7};
        for (int i = 0; i < num_elems; ++i)
        {
            uint4 cvt_elem_pack8 = h_cvt_result[i];
            uint4 opt_elem_pack8  = h_opt_result[i];

            for (int j = 0; j < 8; ++j)
            {
                float cvt_elem = static_cast<float>(reinterpret_cast<half *>(&cvt_elem_pack8)[index_map[j]]);
                float opt_elem = static_cast<float>(reinterpret_cast<half *>(&opt_elem_pack8)[j]);

                if (fabs(cvt_elem - opt_elem) > 0.0001)
                    printf("error compare %d, %d, %f, %f\n", i, j, cvt_elem, opt_elem);
            }
        }

        free(h_cvt_result);
        free(h_opt_result);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run kernel multiple times for averaging
    float total_time = 0;
    int warmup_times = iterations < 100 ? 0 : 100;
    for (int i = 0; i < warmup_times; ++i)
    {
        cudaEventRecord(start);
        test_dequantize<<<num_blocks, threads_per_block>>>(d_data, opt_result, num_elems, opt);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; ++i)
    {
        cudaEventRecord(start);
        test_dequantize<<<num_blocks, threads_per_block>>>(d_data, opt_result, num_elems, opt);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float time;
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    if (opt)
    {
        printf("Kernel with opt: %f ms (average over %d runs)\n", total_time / iterations, iterations);
    }
    else
    {
        printf("Kernel with cvt: %f ms (average over %d runs)\n", total_time / iterations, iterations);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    free(h_data);
    cudaFree(d_data);
    cudaFree(cvt_result);
    cudaFree(opt_result);

    return 0;
}