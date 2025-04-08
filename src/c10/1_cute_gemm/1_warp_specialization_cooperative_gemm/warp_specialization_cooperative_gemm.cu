#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

#include <cutlass/util/command_line.h>

using namespace cute;

namespace c1011
{

// Command line options parsing
struct Options
{
    bool help;

    float alpha, beta;
    int iterations;
    int m, n, k;
    char trans_A, trans_B;

    Options():
        help(false),
        m(5120), n(4096), k(4096),
        alpha(1.f), beta(0.f),
        iterations(1000),
        trans_A('N'),
        trans_B('T')
    { }

    // Parses the command line
    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) {
        help = true;
        return;
        }

        cmd.get_cmd_line_argument("m", m);
        cmd.get_cmd_line_argument("n", n);
        cmd.get_cmd_line_argument("k", k);
        cmd.get_cmd_line_argument("alpha", alpha, 1.f);
        cmd.get_cmd_line_argument("beta", beta, 0.f);
        cmd.get_cmd_line_argument("iterations", iterations);
        cmd.get_cmd_line_argument("trans_A", trans_A);
        cmd.get_cmd_line_argument("trans_B", trans_B);
    }

    /// Prints the usage statement.
    std::ostream & print_usage(std::ostream &out) const {

        out << "cute_gemm\n\n"
        << "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --trans_A=<char>            Trans for A\n\n"
        << "  --trans_B=<char>            Trans for B\n\n"
        << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

        out
        << "\n\nExamples:\n\n"
        << "$ " << "cute_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

        return out;
    }
};

template <class ElementA,
          class ElementB,
          class SmemLayoutA, // (M, K, P)
          class SmemLayoutB> // (N, K, P)

struct SharedStorage
{
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;

    uint64_t tma_barrier[size<2>(SmemLayoutA{})]; // pipe stages
    uint64_t mma_barrier[size<2>(SmemLayoutB{})]; // pipe stages
};

template <class ProblemShape,
          class CtaTiler,
          class TA,
          class SmemLayoutA,
          class TmaA,
          class TB,
          class SmemLayoutB,
          class TmaB,
          class TC,
          class CStride,
          class TiledMma,
          class Alpha,
          class Beta>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(ProblemShape                     shape_mnk,
                 CtaTiler                         cta_tiler,
                 TA const*                        A,
                 CUTLASS_GRID_CONSTANT TmaA const tma_a,
                 TB const*                        B,
                 CUTLASS_GRID_CONSTANT TmaB const tma_b,
                 TC*                              C,
                 CStride                          stride_C,
                 TiledMma                         mma,
                 Alpha                            alpha,
                 Beta                             beta)
{

    // 检查参数
    CUTE_STATIC_ASSERT_V(rank(shape_mnk) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (Tile_M, Tile_N, Tile_K)

    static_assert(is_static<SmemLayoutA>::value);
    static_assert(is_static<SmemLayoutB>::value);

    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // Tile_M
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // Tile_K
    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // Tile_N
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // Tile_K

    // TODO ?
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_mnk), stride_C));         // stride_C for shape MN

    // Full and Tiled Tensors

    // Represent the full tensors
    auto [m, n, k] = shape_mnk;

    Tensor global_gmem_A = tma_a.get_tma_tensor(make_shape(m, k));   // (m, k) tma tensor
    Tensor global_gmem_B = tma_b.get_tma_tensor(make_shape(n, k));   // (n, k) tma tensor
    Tensor global_gmem_C = make_tensor(make_gmem_ptr(C), make_shape(m, n), stride_C); // (M, N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);     // (m, n, k)
    Tensor local_gmem_A = local_tile(global_gmem_A, cta_tiler, cta_coord, Step<_1,  X, _1>{}); // (BLK_M, BLK_K, k)
    Tensor local_gmem_B = local_tile(global_gmem_B, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N, BLK_K, k)
    Tensor local_gmem_C = local_tile(global_gmem_C, cta_tiler, cta_coord, Step<_1, _1,  X>{}); // (BLK_M, BLK_N)

    // Shared memory tensors
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor local_smem_A = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor local_smem_B = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    //
    // Partition the copying of A and B tiles
    //
    // TUTORIAL:
    //   These are TMA partitionings, which have a dedicated custom partitioner.
    //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
    //     Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
    //   The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into ((X,Y),Z)-shaped tensors
    //     with the understanding that the TMA is responsible for everything in mode-0.
    //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
    //
    auto [tAgA, tAsA] = tma_partition(tma_a,
                                      Int<0>{},
                                      Layout<_1>{},
                                      group_modes<0, 2>(local_smem_A),
                                      group_modes<0, 2>(local_gmem_A));
    
    auto [tBgB, tBsB] = tma_partition(tma_b,
                                      Int<0>{},
                                      Layout<_1>{},
                                      group_modes<0, 2>(local_smem_B),
                                      group_modes<0, 2>(local_gmem_B));

    // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
    constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
                                         CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);


    //
    // PREFETCH
    //
    auto kPipeMax = size<1>(tAsA);

    // Total count of tiles
    int k_tile_count = size<1>(tAgA);

    // current tile index in gmem to read from
    int k_tile = 0;

    // Initialize Barriers
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA

    CUTE_UNROLL
    for (int pipe = 0; pipe < kPipeMax; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            ProducerBarType::init(&producer_mbar[pipe],   1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }

    // Ensure barrier init is complete on all CTAs
    cluster_sync();

    // Start async loads for all pipes
    CUTE_UNROLL
    for (int pipe = 0; pipe < kPipeMax; ++pipe)
    {
        if ((warp_idx == 0) && lane_predicate)
        {
            // set expected Tx bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    //
    // Define A/B partitioning and C accumulators
    //
    // TUTORIAL:
    //   The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as views of SMEM.
    //   The MMA Descriptor generation is automatic via inspection and validation of the SMEM Layouts.
    //   Because the MMA reads directly from SMEM and the fragments are descriptors rather than registers,
    //     there is no need for copy(tCsA, tCrA) in the mainloop.
    //
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA    = thr_mma.partition_A(local_smem_A);            // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCsB    = thr_mma.partition_B(local_smem_B);            // (MMA, MMA_N, MMA_K, PIPE)
    Tensor tCgC    = thr_mma.partition_C(local_gmem_C);            // (MMA, MMA_M, MMA_N)

    // Allocate accumulators and clear them
    Tensor tCrC    = thr_mma.make_fragment_C(tCgC);                // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // Allocate "fragments"
    Tensor tCrA    = thr_mma.make_fragment_A(tCsA);                // (MMA, MMA_M, MMA_K, PIPE)
    Tensor tCrB    = thr_mma.make_fragment_B(tCsB);                // (MMA, MMA_N, MMA_K, PIPE)

    //
    // Pipelined Main Loop
    //
    // Tutorial:
    //     Rather than interleaving the stages and instructions like in SM70 and SM80
    //     the SM90 mainloops rely on explicit producer-consumer synchronization
    //     on the purely async instructions TMA and MMA
    //
    //     More advanced pipeline and warp-specialization strategies are available in CUTLASS mainloops

    // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
    // that flips each cycle kPipeMax
    auto write_state = cutlass::PipelineState<kPipeMax>();   // TMA writes
    auto read_state = cutlass::PipelineState<kPipeMax>();    // MMA reads

    CUTE_NO_UNROLL
    while (k_tile_count > -kPipeMax)
    {
        // Wait for Producer to complete
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        // MMAs to cover 1 K_Tile
        warpgroup_arrive();
        // (V, M) x (V, N) => (V, M, N)
        gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
        warpgroup_commit_batch();

        // Wait for all MMAs in a K_Tile to complete
        warpgroup_wait<0>();

        // Notify that consumption is done
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if ((warp_idx == 0) && lane_predicate)
        {
            int pipe = write_state.index();
            // Wait for consumer to complete consumption
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            // Set expected Tx Bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);

            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }

        --k_tile_count;
        ++k_tile;
    }

    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

template <class TA,
          class TB,
          class TC,
          class Alpha,
          class Beta>
void gemm_nt(int          m,
             int          n,
             int          k,
             Alpha        alpha,
             TA const*    A,
             int          lda,
             TB const*    B,
             int          ldb,
             Beta         beta,
             TC*          C,
             int          ldc,
             cudaStream_t stream = 0)
{
    // define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto problem_shape = make_shape(M, N, K);       // (M, N, K)

    // define NT strides (mixed)
    auto stride_A = make_stride(Int<1>{}, lda);
    auto stride_B = make_stride(Int<1>{}, ldb);
    auto stride_C = make_stride(Int<1>{}, ldc);

    // define CTA tile sizes (static)
    auto tile_M = Int<128>{};
    auto tile_N = Int<128>{};
    auto tile_K = Int<64>{};

    auto cta_tiler = make_shape(tile_M, tile_N, tile_K);

    auto pipeline_stages = Int<3>{}; // pipeline stage nums

    // define the smem layouts
    auto smem_A = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(tile_M, tile_K, pipeline_stages));
    auto smem_B = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(tile_N, tile_K, pipeline_stages));

    // Define the MMA
    // make_tile_mma 有两个参数，AtomThrLayout和PermuteThrLayout，分别设置确认其影响
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

    // Define the TMAs
    // Create Global Memory tensors from TMA inspection
    // 使用TMA的方式有很多，cuda/cute/cutlass分别有啥区别
    Tensor gmem_A = make_tensor(A, make_shape(M, K), stride_A);
    Tensor gmem_B = make_tensor(B, make_shape(N, K), stride_B);

    // Create TMA Atoms with the desired copy operation on the source and destination
    Copy_Atom tma_A = make_tma_atom(SM90_TMA_LOAD{}, gmem_A, smem_A(_, _, 0), make_shape(tile_M, tile_K));
    Copy_Atom tma_B = make_tma_atom(SM90_TMA_LOAD{}, gmem_B, smem_B(_, _, 0), make_shape(tile_N, tile_K));

    //
    // Setup and Launch
    //
    // Launch parameter setup
    int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(smem_A), decltype(smem_B)>));
    dim3 dim_block(size(tiled_mma));
    dim3 dim_cluster(2, 1, 1);
    dim3 dim_grid(round_up(size(ceil_div(m, tile_M)), dim_cluster.x),
                  round_up(size(ceil_div(n, tile_N)), dim_cluster.y));
    
    cutlass::ClusterLaunchParams params = {dim_grid, dim_block, dim_cluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const *>(&gemm_device<decltype(problem_shape),
                                                                         decltype(cta_tiler),
                                                                         TA,
                                                                         decltype(smem_A),
                                                                         decltype(tma_A),
                                                                         TB,
                                                                         decltype(smem_B),
                                                                         decltype(tma_B),
                                                                         TC,
                                                                         decltype(stride_C),
                                                                         decltype(tiled_mma),
                                                                         decltype(alpha),
                                                                         decltype(beta)>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
                                                               kernel_ptr,
                                                               problem_shape,
                                                               cta_tiler,
                                                               A,
                                                               tma_A,
                                                               B,
                                                               tma_B,
                                                               C,
                                                               stride_C,
                                                               tiled_mma,
                                                               alpha,
                                                               beta);

    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}


template <class TA,
          class TB,
          class TC,
          class Alpha,
          class Beta>
void gemm_tn(int          m,
             int          n,
             int          k,
             Alpha        alpha,
             TA const*    A,
             int          lda,
             TB const*    B,
             int          ldb,
             Beta         beta,
             TC*          C,
             int          ldc,
             cudaStream_t stream = 0)
{
    // define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    auto problem_shape = make_shape(M, N, K);       // (M, N, K)

    // define TN strides (mixed)
    auto stride_A = make_stride(lda, Int<1>{});
    auto stride_B = make_stride(ldb, Int<1>{});
    auto stride_C = make_stride(Int<1>{}, ldc);

    // define CTA tile sizes (static)
    auto tile_M = Int<128>{};
    auto tile_N = Int<128>{};
    auto tile_K = Int<64>{};

    auto cta_tiler = make_shape(tile_M, tile_N, tile_K);

    auto pipeline_stages = Int<3>{}; // pipeline stage nums

    // define the smem layouts
    auto smem_A = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(tile_M, tile_K, pipeline_stages));
    auto smem_B = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(tile_N, tile_K, pipeline_stages));

    // Define the MMA
    // make_tile_mma 有两个参数，AtomThrLayout和PermuteThrLayout，分别设置确认其影响
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

    // Define the TMAs
    // Create Global Memory tensors from TMA inspection
    // 使用TMA的方式有很多，cuda/cute/cutlass分别有啥区别
    Tensor gmem_A = make_tensor(A, make_shape(M, K), stride_A);
    Tensor gmem_B = make_tensor(B, make_shape(N, K), stride_B);

    // Create TMA Atoms with the desired copy operation on the source and destination
    Copy_Atom tma_A = make_tma_atom(SM90_TMA_LOAD{}, gmem_A, smem_A(_, _, 0), make_shape(tile_M, tile_K));
    Copy_Atom tma_B = make_tma_atom(SM90_TMA_LOAD{}, gmem_B, smem_B(_, _, 0), make_shape(tile_N, tile_K));

    //
    // Setup and Launch
    //
    // Launch parameter setup
    int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(smem_A), decltype(smem_B)>));
    dim3 dim_block(size(tiled_mma));
    dim3 dim_cluster(2, 1, 1);
    dim3 dim_grid(round_up(size(ceil_div(m, tile_M)), dim_cluster.x),
                  round_up(size(ceil_div(n, tile_N)), dim_cluster.y));
    
    cutlass::ClusterLaunchParams params = {dim_grid, dim_block, dim_cluster, smem_size};

    void const* kernel_ptr = reinterpret_cast<void const *>(&gemm_device<decltype(problem_shape),
                                                                         decltype(cta_tiler),
                                                                         TA,
                                                                         decltype(smem_A),
                                                                         decltype(tma_A),
                                                                         TB,
                                                                         decltype(smem_B),
                                                                         decltype(tma_B),
                                                                         TC,
                                                                         decltype(stride_C),
                                                                         decltype(tiled_mma),
                                                                         decltype(alpha),
                                                                         decltype(beta)>);
    
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
                                                               kernel_ptr,
                                                               problem_shape,
                                                               cta_tiler,
                                                               A,
                                                               tma_A,
                                                               B,
                                                               tma_B,
                                                               C,
                                                               stride_C,
                                                               tiled_mma,
                                                               alpha,
                                                               beta);
    
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}

template <class TA,
          class TB,
          class TC,
          class Alpha,
          class Beta>
void gemm(char      transA,
          char      transB,
          int       m,
          int       n,
          int       k,
          Alpha     alpha,
          TA const* A,
          int       lda,
          TB const* B,
          int       ldb,
          Beta      beta,
          TC*       C,
          int       ldc,
          cudaStream_t stream = 0)
{
    if (transA == 'N' && transB == 'T')
    {
        return gemm_nt(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
    }
    else
    {
        return gemm_tn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
    }
    assert(false && "Not implemented");
}

}

int main(int argc, char const** args)
{
    // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
    // and must have compute capability at least 90.

    if (__CUDACC_VER_MAJOR__ < 12)
    {
        std::cerr << "This example requires CUDA 12 or newer. \n";
        return 0;
    }

    cudaDeviceProp props;
    int current_device_id;
    CUTE_CHECK_ERROR(cudaGetDevice(&current_device_id));
    CUTE_CHECK_ERROR(cudaGetDeviceProperties(&props, current_device_id));

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (props.major != 9 || props.minor != 0) {
      std::cerr
        << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
      return 0;
    }

    //
    // Parse options
    //

    c1010::Options options;

    options.parse(argc, args);

    if (options.help) {
      options.print_usage(std::cout) << std::endl;
      return 0;
    }

    //
    // Evaluate CUTLASS kernels
    //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TI = cute::half_t;

    TI alpha = TI(options.alpha);
    TI beta  = TI(options.beta);

    int m = options.m;
    int n = options.n;
    int k = options.k;

    char trans_A = options.trans_A;
    char trans_B = options.trans_B;

    int iterations = options.iterations;

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    // Initialize the tensors
    for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n*k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    double gflops = (2.0*m*n*k) * 1e-9;
    
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;

    if (trans_A == 'N')
    {
        ldA = m;
    }
    else if (trans_A == 'T')
    {
        ldA = k;
    }
    else
    {
        assert(false);
    }

    if (trans_B == 'N')
    {
        ldB = k;
    }
    else if (trans_B == 'T')
    {
        ldB = n;
    }
    else
    {
        assert(false);
    }

    // Run once
    d_C = h_C;
    c1011::gemm(trans_A, trans_B,
                m, n, k,
                alpha,
                d_A.data().get(), ldA,
                d_B.data().get(), ldB,
                beta,
                d_C.data().get(), ldC);
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> cute_result = d_C;

    // Timing iterations
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        c1011::gemm(trans_A, trans_B, m, n, k,
                    alpha,
                    d_A.data().get(), ldA,
                    d_B.data().get(), ldB,
                    beta,
                    d_C.data().get(), ldC);
    }
    double cute_time = timer.seconds() / iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
#endif

    return 0;
}