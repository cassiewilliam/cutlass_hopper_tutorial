
#pragma once

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include <cute/tensor.hpp>
#include <cutlass/conv/convolution.h>
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include <cutlass/util/packed_stride.hpp>

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

#include "src/utils/logger.h"
#include "hopper_gemm_ws.h"
#include "kernel_traits.h"
#include "tile_scheduler.hpp"
#include "mainloop_sm90_tma_ws.hpp"
#include "epilogue_sm90_tma_ws.hpp"

namespace c1000
{

template <typename T>
void HopperGemmWsRunner<T>::gemm(void*        D,
                                 void* const* A,
                                 void* const* B,
                                 void* const* C,
                                 int          m,
                                 int          n,
                                 int          k,
                                 float        scale_A,
                                 float        scale_B,
                                 char*        workspace,
                                 size_t       workspace_bytes,
                                 cudaStream_t stream,
                                 int*         occupancy)
{
    // kBlockM_,
    // kBlockN_,
    // kBlockK_,
    // kWarps_,
    // kStages_,
    // kClusterM_  = 1,
    // kClusterN_  = 1,
    // Element_    = cutlass::half_t,
    // OutputType_ = Element_,
    // FP32Accum   = true>
    using KernelTraits = KernelTraits<256, 192, 128, 12, 2, 1, 1, T>;
    
    auto smem_layout_A = typename KernelTraits::SmemLayoutA{};
    auto smem_layout_B = typename KernelTraits::SmemLayoutB{};
    auto smem_layout_C = typename KernelTraits::SmemLayoutC{};

    using TileShape_MNK = typename KernelTraits::TileShape_MNK;
    using ClusterShape_MNK = typename KernelTraits::ClusterShape_MNK;

    if (k % KernelTraits::kBlockK != 0)
    {
        print("k not divisible by kBlockK = %d\n", Kernel_traits::kBlockK);
        return;
    }

    using CollectiveMainloop = CollectiveMainloop<KernelTraits>;
    using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
    using Scheduler = SingleTileScheduler;

    typename CollectiveMainloop::Params mainloop_params = 
        CollectiveMainloop::to_underlying_arguments({
            A,
            make_layout(make_shape(M, K), make_stride(ldA, Int<1>{})), // layout A
            B,
            make_layout(make_shape(N, K), make_stride(ldB, Int<1>{})), // layout B
        });
    
    typename CollectiveEpilogue::Params epilogue_params = 
        CollectiveEpilogue::to_underlying_arguments({
            C,
            make_layout(make_shape(M, N), make_stride(ldC, Int<1>{}))  // layout C
        });

    int num_blocks_m = cutlass::ceil_div(m, KernelTraits::kBlockM);
    int num_blocks_n = cutlass::ceil_div(n, KernelTraits::kBlockN);
    int num_blocks_k = cutlass::ceil_div(k, KernelTraits::kBlockK);

    // round if using clusters, 为什么需要单独round clusters，cluster读整体运行的影响 ？
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    num_blocks_n = cutlass::ceil_div(num_blocks_n, size<1>(ClusterShape{})) * size<1>(ClusterShape{});

    int device;
    cudaGetDevice(&device);
    CUTE_CHECK_LAST();
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    CUTE_CHECK_LAST();

    auto workspace_size = Scheduler::get_workspace_size(num_blocks_m * num_blocks_n, num_sms);
    CUTE_CHECK_LAST();

    typename Scheduler::Arguments scheduler_args = {num_blocks_m, num_blocks_n, num_blocks_k, workspace};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    // Get the ptr to kernel function
    void* kernel;
    // TODO: 给定kernel function

    int smem_size=  sizeof(typename KernelTraits::SharedStorage);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        CUTE_CHECK_LAST();
    }

    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, num_sms);
    static constexpr int cta_size = KernelTraits::kNWarps * cutlass::NumThreadsPerWarp;
    dim3 block_dims(cta_size);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};

    // TODO: 可打印启动参数
    cutlass::launch_kernel_on_cluster(
        launch_params,
        kernel,
        mainloop_params,
        epilogue_params,
        scheduler_params
    );


    GPU_Clock timer;

    if (iters > 0) {
        timer.start();
        for (int i = 0; i < iters; ++i) {
            Scheduler::initialize_workspace(num_blocks_m * num_blocks_n, num_sms, workspace, stream);

            cutlass::launch_kernel_on_cluster(
                launch_params, kernel,
                mainloop_params, epilogue_params, scheduler_params);
        }
        int worktiles = (M / Kernel_traits::kBlockM) * (N / Kernel_traits::kBlockN);
        int waves = (worktiles + num_sms - 1) / num_sms;
        int partial_wave_size = worktiles % num_sms;

        double avg_time = timer.seconds() / iters;
        double tflops = (2.0 * M * N * K) * 1e-12;
        if (csv)
            print("%d,%d,%d,%d,%d,%d,%.4f,%.1f,%d\n",
                  M, N, K, worktiles, waves, partial_wave_size, avg_time * 1000, tflops / avg_time, iters);
        else {
            print("TN GEMM, M x N x K = %d x %d x %d\n", M, N, K);
            print("Time:     [%6.1f]TFlop/s  %6.4fms  (average of %d iterations)\n",
                  tflops / avg_time, avg_time*1000, iters);
        }
    }
    CUTE_CHECK_LAST();
    cudaFree(workspace);
    CUTE_CHECK_LAST();
}

}