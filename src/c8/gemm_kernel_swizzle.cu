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

// reference from https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/common/ptx.h
#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
                 : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                                                                              \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3))

using namespace cutlass;
using namespace cute;

// 3. HGEMM with Tensor Core wmma PTX API
__global__ void mma_m16n8k16_kernel(bool  A_transpose,
                                    bool  B_transpose,
                                    int   m,
                                    int   n,
                                    int   k,
                                    float alpha,
                                    const half *__restrict__ A,
                                    const half *__restrict__ B,
                                    float beta,
                                    const float *__restrict__ C,
                                    float *__restrict__ D)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int WARP_SIZE = 32;

    const int lda = k;
    const int ldb = k;
    const int ldc = n;

    const int warp_row_index = blockIdx.x;
    const int warp_col_index = blockIdx.y;

    // NOTE: warp对应矩阵M方向实际的index位置
    const int warp_m_offset = warp_row_index * MMA_M;
    // NOTE: warp对应矩阵N方向实际的index位置
    const int warp_n_offset = warp_col_index * MMA_N;

    if (warp_m_offset >= m || warp_n_offset >= n)
    {
        return;
    }

    __shared__ half  A_shmem[MMA_M][MMA_K];
    __shared__ half  B_shmem[MMA_N][MMA_K];
    __shared__ float C_shmem[MMA_M][MMA_N];
    __shared__ float D_shmem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RAcc[4] = {0, 0, 0, 0};
    float    RC[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (size_t i = 0; i < k; i += MMA_K)
    {
        // Step1: A and B Global To Shared Memory
        *((float4 *)(&A_shmem[lane_id / 2][(lane_id % 2) * 8])) =
            *((float4 *)(A + (warp_m_offset + lane_id / 2) * lda + i + (lane_id % 2) * 8));

        if (lane_id < MMA_N * 2)
        {
            *((float4 *)(&B_shmem[lane_id / 2][(lane_id % 2) * 8])) =
                *((float4 *)(B + (warp_n_offset + lane_id / 2) * ldb + i + (lane_id % 2) * 8));
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        // Step2: A and B Global To Shared Memory with ldmatrix
        uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&A_shmem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_shmem_lane_addr);

        uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&B_shmem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_shmem_lane_addr);

        // Step3: TensorCore计算矩阵乘法
        HMMA16816(RAcc[0], RAcc[1], RAcc[2], RAcc[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RAcc[0], RAcc[1], RAcc[2], RAcc[3]);

        __syncthreads();
    }

    // Step4: 加载C Global => Shared => Register 并且进行计算
    *((float4 *)(&C_shmem[lane_id / 2][0])) = *((float4 *)(&C[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 0]));
    *((float4 *)(&C_shmem[lane_id / 2][4])) = *((float4 *)(&C[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 4]));

    // Shared To Register
    RC[0] = C_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 0];
    RC[1] = C_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 1];
    RC[2] = C_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 0];
    RC[3] = C_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 1];

    // NOTE: D = α * (A @ B) + β * C
    for (int i = 0; i < 4; ++i)
    {
        RC[i] = alpha * ((float *)RAcc)[i] + beta * RC[i];
    }

    // Step5: 输出Register File To Shared Memory
    D_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 0] = RC[0];
    D_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 1] = RC[1];
    D_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 0] = RC[2];
    D_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 1] = RC[3];

    __syncthreads();

    // Step6. 结果写出，每次写出128bit 4个float数据，一行分两次写出
    *((float4 *)(&D[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 0])) = *((float4 *)(&D_shmem[lane_id / 2][0]));
    *((float4 *)(&D[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 4])) = *((float4 *)(&D_shmem[lane_id / 2][4]));
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

void vis_swizzle_layout()
{
    auto sw_layout0 = Layout<Shape<_8, _8>, Stride<_8, _1>>{};
    auto sw_layout1 = composition(Swizzle<3, 0, 3>{}, Layout<Shape<_8, _8>, Stride<_8, _1>>{});
    auto sw_layout2 = composition(Swizzle<3, 0, -3>{}, Layout<Shape<_8, _8>, Stride<_8, _1>>{});
    
    print_layout(sw_layout0);
    print_layout(sw_layout1);
    print_layout(sw_layout2);
}

/// Program entrypoint
int main(int argc, char const** argv)
{

    vis_swizzle_layout();
    /*
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
    */
    return 0;
}