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

// CUDA kernel function
__global__ void ThreadGroupTilingX(const uint2 dispatchGridDim,    // Dispatch grid dimensions
                                   const uint2 ctaDim,             // Block dimensions (thread block size)
                                   const uint  maxTileWidth,       // Maximum tile width
                                   uint2*      swizzledThreadID)   // Output swizzled thread IDs
{
    // Thread and block indices
    uint2 groupThreadID = {threadIdx.x, threadIdx.y};
    uint2 groupId = {blockIdx.x, blockIdx.y};

    // A perfect tile has dimensions [maxTileWidth, dispatchGridDim.y]
    const uint Number_of_CTAs_in_a_perfect_tile = maxTileWidth * dispatchGridDim.y;

    // Number of perfect tiles
    const uint Number_of_perfect_tiles = dispatchGridDim.x / maxTileWidth;

    // Total CTAs in all perfect tiles
    const uint Total_CTAs_in_all_perfect_tiles = Number_of_perfect_tiles * maxTileWidth * dispatchGridDim.y;

    // Flattened thread group ID
    const uint vThreadGroupIDFlattened = dispatchGridDim.x * groupId.y + groupId.x;

    // Determine tile and local CTA ID
    const uint Tile_ID_of_current_CTA = vThreadGroupIDFlattened / Number_of_CTAs_in_a_perfect_tile;
    const uint Local_CTA_ID_within_current_tile = vThreadGroupIDFlattened % Number_of_CTAs_in_a_perfect_tile;

    uint Local_CTA_ID_y_within_current_tile;
    uint Local_CTA_ID_x_within_current_tile;

    if (Total_CTAs_in_all_perfect_tiles <= vThreadGroupIDFlattened)
    {
        // Handle imperfect tiles
        uint X_dimension_of_last_tile = dispatchGridDim.x % maxTileWidth;
        X_dimension_of_last_tile = max(1u, X_dimension_of_last_tile);
        Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / X_dimension_of_last_tile;
        Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % X_dimension_of_last_tile;
    }
    else
    {
        Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / maxTileWidth;
        Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % maxTileWidth;
    }

    // Swizzle flattened thread group ID
    const uint Swizzled_vThreadGroupIDFlattened =
        Tile_ID_of_current_CTA * maxTileWidth +
        Local_CTA_ID_y_within_current_tile * dispatchGridDim.x +
        Local_CTA_ID_x_within_current_tile;

    uint2 SwizzledvThreadGroupID;
    SwizzledvThreadGroupID.y = Swizzled_vThreadGroupIDFlattened / dispatchGridDim.x;
    SwizzledvThreadGroupID.x = Swizzled_vThreadGroupIDFlattened % dispatchGridDim.x;

    // Calculate swizzled thread IDs
    uint2 SwizzledvThreadID;
    SwizzledvThreadID.x = ctaDim.x * SwizzledvThreadGroupID.x + groupThreadID.x;
    SwizzledvThreadID.y = ctaDim.y * SwizzledvThreadGroupID.y + groupThreadID.y;

    // Store result
    swizzledThreadID[blockIdx.y * gridDim.x + blockIdx.x] = SwizzledvThreadID;
}

template <typename ElementT, typename LayoutB, typename ElementO>
struct Buffers
{
    cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_a;
    cutlass::HostTensor<ElementT, LayoutB> tensor_b;
    cutlass::HostTensor<ElementO, cutlass::layout::RowMajor> tensor_d;
    cutlass::HostTensor<ElementO, cutlass::layout::RowMajor> tensor_ref_d;
    cutlass::HostTensor<ElementO, cutlass::layout::RowMajor> tensor_c_bias;
};

/// Program entrypoint
int main(int argc, char const** argv)
{

    // Current device must must have compute capability at least 80
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);
    if (!((props.major * 10 + props.minor) >= 90))
    {
        std::cerr << "Hopper Tensor Core operations must be run on a machine with compute capability at least 90."
                  << std::endl;

        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        exit(0);
    }

    using ElementT = cutlass::half_t;
    using ElementAccumulatorT = float;
    using ElementComputeT = float;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementO = float;

    Buffers<ElementT, LayoutB, ElementO> buffers;

    // Parse commandline options
    Options options("hopper_mma_gemm_raw");
    options.parse(argc, argv);

    if (options.help)
    {
        options.print_usage(std::cout) << std::endl;
        exit(0);
    }

    std::cout << options.iterations << " timing iterations of " << options.problem_size.m() << " x "
              << options.problem_size.n() << " x " << options.problem_size.k() << " matrix-matrix multiply"
              << std::endl;

    if (!options.valid())
    {
        std::cerr << "Invalid problem." << std::endl;
        exit(-1);
    }

    //
    // Initialize GEMM datasets
    //

    // Initialize tensors using CUTLASS helper functions
    buffers.tensor_a.resize(options.problem_size.mk());          // <- Create matrix A with dimensions M x K
    buffers.tensor_b.resize(options.problem_size.kn());          // <- Create matrix B with dimensions K x N
    buffers.tensor_c_bias.resize({1, options.problem_size.n()}); // <- Create broadcast vector with dimensions 1 x N
    buffers.tensor_d.resize(options.problem_size.mn());          // <- Create matrix D with dimensions M x N used to store output from test kernel
    buffers.tensor_ref_d.resize(options.problem_size.mn());      // <- Create matrix D with dimensions M x N used to store output from reference kernel

    int _init_bits = options.real ? -1 : 0;

    // Fill matrix A on host with uniform-random data [-2, 2]
    if (options.debug)
    {
        cutlass::Array<ElementT, 2> range;
        range[0] = ElementT(256);
        range[1] = ElementT(1);
        cutlass::reference::host::TensorFillLinear(buffers.tensor_a.host_view(), range);
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_a.host_view(), 1, ElementT(2), ElementT(-2), _init_bits);
    }

    // Fill matrix B on host with uniform-random data [-2, 2]
    if (options.debug)
    {
        cutlass::reference::host::TensorFillIdentity(buffers.tensor_b.host_view());
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_b.host_view(), 1, ElementT(2), ElementT(-2), _init_bits);
    }

    if (options.debug || !options.has_bias)
    {
        cutlass::reference::host::TensorFill(buffers.tensor_c_bias.host_view());
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_c_bias.host_view(), 1, ElementO(2), ElementO(-2), _init_bits);
    }

    if (options.debug)
    {
        std::cout << "A=" << std::endl << buffers.tensor_a.host_view() << std::endl;
        std::cout << "B=" << std::endl << buffers.tensor_b.host_view() << std::endl;
        std::cout << "C=" << std::endl << buffers.tensor_c_bias.host_view() << std::endl;
    }

    //
    // Compute reference output
    //

    // Copy data from host to GPU
    buffers.tensor_a.sync_device();
    buffers.tensor_b.sync_device();
    buffers.tensor_c_bias.sync_device();

   // Zero-initialize reference output matrix D
    cutlass::reference::host::TensorFill(buffers.tensor_ref_d.host_view());
    buffers.tensor_ref_d.sync_device();

    cutlass::reference::host::TensorFill(buffers.tensor_d.host_view());
    buffers.tensor_d.sync_device();

    // Create instantiation for device reference gemm kernel
    // Reference device GEMM implementation type
    using DeviceGemmReference = cutlass::reference::device::Gemm<ElementT, 
                                                                 cutlass::layout::RowMajor,
                                                                 ElementT,
                                                                 LayoutB,
                                                                 ElementO,
                                                                 cutlass::layout::RowMajor,
                                                                 ElementAccumulatorT,
                                                                 ElementAccumulatorT>;
    // Create instantiation for device reference gemm kernel
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(options.problem_size, 
                   ElementAccumulatorT(options.alpha), 
                   buffers.tensor_a.device_ref(),
                   buffers.tensor_b.device_ref(),
                   ElementAccumulatorT(options.beta),
                   buffers.tensor_ref_d.device_ref(),
                   buffers.tensor_ref_d.device_ref());

    // Wait for kernels to finish
    cudaDeviceSynchronize();

    // Copy output data from reference kernel to host for comparison
    buffers.tensor_ref_d.sync_host();

    // Add broadcast vector (without multiplier)
    // Vector broadcast on host
    for (int i = 0; i < options.problem_size.m(); ++i)
    {
        for (int j = 0; j < options.problem_size.n(); ++j)
        {
            buffers.tensor_ref_d.host_view().ref().at({i, j}) += buffers.tensor_c_bias.host_view().ref().at({0, j});
        }
    }

    cudaDeviceSynchronize();

    if (options.debug)
    {
        std::cout << "tensor_ref_d=" << buffers.tensor_ref_d.host_view() << std::endl;
    }

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 8;
    constexpr int WARP_SIZE = 32;

    dim3 block_dim(WARP_SIZE);
    dim3 grid_dim(UP_DIV(options.problem_size.m(), (WMMA_M * block_dim.x / WARP_SIZE)), UP_DIV(options.problem_size.n(), (WMMA_N * block_dim.y)));

    mma_m16n8k16_kernel<<<grid_dim, block_dim>>>
    (
        false,
        false,
        options.problem_size.m(),
        options.problem_size.n(),
        options.problem_size.k(),
        options.alpha,
        reinterpret_cast<const half *>(buffers.tensor_a.device_data()),
        reinterpret_cast<const half *>(buffers.tensor_b.device_data()),
        options.beta,
        reinterpret_cast<float *>(buffers.tensor_c_bias.device_data()),
        reinterpret_cast<float *>(buffers.tensor_d.device_data())
    );

    // Copy output data from CUTLASS and reference kernel to host for comparison
    buffers.tensor_d.sync_host();

    // Initialize Result
    Result result;

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    {
        if (!options.no_check)
        {
            result.passed = cutlass::reference::host::TensorRelativelyEquals(
                buffers.tensor_d.host_view(), buffers.tensor_ref_d.host_view(), ElementO{1e-3}, ElementO{1e-3});

            // EXPECT_TRUE(result.passed);

            double err = cutlass::reference::host::TensorRelativeErrorMetric(
                buffers.tensor_d.host_view(), buffers.tensor_ref_d.host_view());

            std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << " \t Relative error: " << err
                        << std::endl;

            if (!result.passed)
            {
                for (int i = 0; i < options.problem_size.m(); ++i)
                {
                    for (int j = 0; j < options.problem_size.n(); ++j)
                    {
                        printf("index: %d, %d, %.5f, %.5f, %.5f\n", i, j, 
                            buffers.tensor_ref_d.host_view().ref().at({i, j}),
                            buffers.tensor_d.host_view().ref().at({i, j}),
                            buffers.tensor_ref_d.host_view().ref().at({i, j}) - buffers.tensor_d.host_view().ref().at({i, j}));
                    }
                }
            }
        }
    }

    // Run profiling loop
    {
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        if (options.iterations > 0)
        {
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            for (int iter = 0; iter < options.iterations; ++iter)
            {
                mma_m16n8k16_kernel<<<grid_dim, block_dim>>>
                (
                    false,
                    false,
                    options.problem_size.m(),
                    options.problem_size.n(),
                    options.problem_size.k(),
                    options.alpha,
                    reinterpret_cast<const half *>(buffers.tensor_a.device_data()),
                    reinterpret_cast<const half *>(buffers.tensor_b.device_data()),
                    options.beta,
                    reinterpret_cast<float *>(buffers.tensor_c_bias.device_data()),
                    reinterpret_cast<float *>(buffers.tensor_d.device_data())
                );
            }
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
   
            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);

            result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
            result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

            std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
            std::cout << "  GFLOPs: " << result.gflops << std::endl;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
