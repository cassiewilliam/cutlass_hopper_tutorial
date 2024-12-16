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

// Divide the 2D Dispatch Grid into tiles of dimension [N, dispatch_grid_dim.y]
__device__ uint2 ThreadGroupTilingX(const uint2 dispatch_grid_dim, // Arguments of the Dispatch call
                                    const uint2 cta_dim,           // Known thread dimensions in CUDA, e.g., dim3(8, 8, 1)
                                    const uint max_tile_width,     // User parameter (N). Recommended values: 8, 16, or 32.
                                    const uint2 group_thread_id,   // Local thread ID within a thread block (threadIdx.x, threadIdx.y)
                                    const uint2 group_id)          // Block ID (blockIdx.x, blockIdx.y)
{
    // A perfect tile is one with dimensions = [max_tile_width, dispatch_grid_dim.y]
    const uint Number_of_CTAs_in_a_perfect_tile = max_tile_width * dispatch_grid_dim.y;

    // Possible number of perfect tiles
    const uint Number_of_perfect_tiles = dispatch_grid_dim.x / max_tile_width;

    // Total number of CTAs present in the perfect tiles
    const uint Total_CTAs_in_all_perfect_tiles = Number_of_perfect_tiles * max_tile_width * dispatch_grid_dim.y;
    const uint vThreadGroupIDFlattened = dispatch_grid_dim.x * group_id.y + group_id.x;

    // Tile ID of current CTA: maps the current CTA to its TILE ID
    const uint Tile_ID_of_current_CTA = vThreadGroupIDFlattened / Number_of_CTAs_in_a_perfect_tile;
    const uint Local_CTA_ID_within_current_tile = vThreadGroupIDFlattened % Number_of_CTAs_in_a_perfect_tile;
    uint Local_CTA_ID_y_within_current_tile;
    uint Local_CTA_ID_x_within_current_tile;

    if (Total_CTAs_in_all_perfect_tiles <= vThreadGroupIDFlattened) {
        // Path taken only if the last tile has imperfect dimensions
        uint X_dimension_of_last_tile = dispatch_grid_dim.x % max_tile_width;
        X_dimension_of_last_tile = max(1u, X_dimension_of_last_tile);

        Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / X_dimension_of_last_tile;
        Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % X_dimension_of_last_tile;
    } else {
        Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / max_tile_width;
        Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % max_tile_width;
    }

    const uint Swizzled_vThreadGroupIDFlattened =
        Tile_ID_of_current_CTA * max_tile_width +
        Local_CTA_ID_y_within_current_tile * dispatch_grid_dim.x +
        Local_CTA_ID_x_within_current_tile;

    uint2 SwizzledvThreadGroupID;
    SwizzledvThreadGroupID.y = Swizzled_vThreadGroupIDFlattened / dispatch_grid_dim.x;
    SwizzledvThreadGroupID.x = Swizzled_vThreadGroupIDFlattened % dispatch_grid_dim.x;

    uint2 SwizzledvThreadID;
    SwizzledvThreadID.x = cta_dim.x * SwizzledvThreadGroupID.x + group_thread_id.x;
    SwizzledvThreadID.y = cta_dim.y * SwizzledvThreadGroupID.y + group_thread_id.y;

    return SwizzledvThreadID;
}

__global__ void KernelExample(uint2 dispatch_grid_dim, uint2 cta_dim, uint max_tile_width, int* data, int num_elems, bool use_swizzle)
{
    uint2 group_thread_id = {threadIdx.x, threadIdx.y};
    uint2 group_id = {blockIdx.x, blockIdx.y};

    uint2 thread_id;
    if (use_swizzle) {
        thread_id = ThreadGroupTilingX(dispatch_grid_dim, cta_dim, max_tile_width, group_thread_id, group_id);
    } else {
        thread_id.x = cta_dim.x * group_id.x + group_thread_id.x;
        thread_id.y = cta_dim.y * group_id.y + group_thread_id.y;
    }

    // Calculate linear thread ID for memory access
    int linear_thread_id = thread_id.y * dispatch_grid_dim.x * cta_dim.x + thread_id.x;

    if (linear_thread_id < num_elems) {
        // Perform memory access to test cache behavior
        int value = data[linear_thread_id];
        data[linear_thread_id] = value + 1; // Simple operation to avoid optimization out
    }
}

int main(int argc, char const** argv)
{
    constexpr int num_elems = 1024 * 1024 * 128; // 128M elements
    int* h_data = (int*)malloc(num_elems * sizeof(int));
    int* d_data;

    // Initialize host data
    for (int i = 0; i < num_elems; ++i) {
        h_data[i] = i;
    }

    bool swizzle = false;
    int iterations = 100;
    cutlass::CommandLine cmd(argc, argv);

    if (cmd.check_cmd_line_flag("swizzle"))
    {
        swizzle = true;
    }

    cmd.get_cmd_line_argument("iterations", iterations);

    cudaMalloc(&d_data, num_elems * sizeof(int));
    cudaMemcpy(d_data, h_data, num_elems * sizeof(int), cudaMemcpyHostToDevice);

    constexpr int grid_x = 32;
    constexpr int grid_y = num_elems / 128 / 32;
    uint2 dispatch_grid_dim = {grid_x, grid_y};
    uint2 cta_dim = {16, 8};
    uint max_tile_width = 16;

    dim3 threads_per_block(cta_dim.x, cta_dim.y);
    dim3 num_blocks((dispatch_grid_dim.x + cta_dim.x - 1) / cta_dim.x, (dispatch_grid_dim.y + cta_dim.y - 1) / cta_dim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run kernel multiple times for averaging
    float total_time = 0;
    int warmup_times = iterations < 100 ? 0 : 100;
    for (int i = 0; i < warmup_times; ++i) {
        cudaEventRecord(start);
        KernelExample<<<num_blocks, threads_per_block>>>(dispatch_grid_dim, cta_dim, max_tile_width, d_data, num_elems, swizzle);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start);
        KernelExample<<<num_blocks, threads_per_block>>>(dispatch_grid_dim, cta_dim, max_tile_width, d_data, num_elems, swizzle);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float time;
        cudaEventElapsedTime(&time, start, stop);
        total_time += time;
    }
    if (swizzle)
    {
        printf("Kernel with    Swizzle: %f ms (average over %d runs)\n", total_time / iterations, iterations);
    }
    else
    {
        printf("Kernel without Swizzle: %f ms (average over %d runs)\n", total_time / iterations, iterations);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy data back to host and validate
    cudaMemcpy(h_data, d_data, num_elems * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("Data[%d] = %d\n", i, h_data[i]);
    }

    // Cleanup
    free(h_data);
    cudaFree(d_data);

    return 0;
}