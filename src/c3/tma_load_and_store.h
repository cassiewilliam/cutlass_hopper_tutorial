#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cutlass/numeric_types.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/command_line.h>
#include <cutlass/util/helper_cuda.hpp>
#include <cutlass/util/print_error.hpp>

#include <cutlass/detail/layout.hpp>

namespace c3
{

inline void set_smem_size(int smem_size, void const *kernel)
{
  // account for dynamic smem capacity if needed
  if (smem_size >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      std::cout << "  Shared Memory Allocation Failed " << std::endl
                << " cudaFuncSetAttribute() returned error: "
                << cudaGetErrorString(result) << std::endl;
    }
  }
}

template <class Element, class SmemLayout>
struct SharedStorageTMA
{
    cute::array_aligned<Element, 
                        cute::cosize_v<SmemLayout>,
                        cutlass::detail::alignment_for_swizzle(SmemLayout{})> smem;

    cutlass::arch::ClusterTransactionBarrier mbarrier;
};

template <typename TiledCopyS_,
          typename TiledCopyD_,
          typename GmemLayout_,
          typename SmemLayout_,
          typename TileShape_>
struct Params
{
    using TiledCopyS = TiledCopyS_;
    using TiledCopyD = TiledCopyD_;
    using GmemLayout = GmemLayout_;
    using SmemLayout = SmemLayout_;
    using TileShape  = TileShape_;

    TiledCopyS const tma_load;
    TiledCopyD const tma_store;
    GmemLayout const gmem_layout;
    SmemLayout const smem_layout;
    TileShape  const tile_shape;

    Params(TiledCopyS const& tma_load,
           TiledCopyD const& tma_store,
           GmemLayout const& gmem_layout,
           SmemLayout const& smem_layout,
           TileShape  const& tile_shape)
        : tma_load(tma_load)
        , tma_store(tma_store)
        , gmem_layout(gmem_layout)
        , smem_layout(smem_layout)
        , tile_shape(tile_shape) 
        {

        }
};

template <int   kNumThreads,
          class Element,
          class Params>
__global__ static void __launch_bounds__(kNumThreads, 1)
tma_load_store_kernel(CUTE_GRID_CONSTANT Params const params)
{
    using namespace cute;

    // Get Layouts and tiled copies from Params struct
    using GmemLayout = typename Params::GmemLayout;
    using SmemLayout = typename Params::SmemLayout;
    using TileShape  = typename Params::TileShape;

    auto& tma_load = params.tma_load;
    auto& tma_store = params.tma_store;
    auto& gmem_layout = params.gmem_layout;
    auto& smem_layout = params.smem_layout;
    auto& tile_shape = params.tile_shape;

    // use shared storage structure to allocate aligned SMEM addresses
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorageTMA<Element, SmemLayout>;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    // define smem tensor
    Tensor smem_tensor = make_tensor(make_smem_ptr(shared_storage.smem.data()), smem_layout);

    // get mbarrier object and its value type
    auto &mbarrier = shared_storage.mbarrier;
    using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
    static_assert(cute::is_same_v<BarrierType, uint64_t>, "value type of mbarrier is uint64_t");

    // constants used for TMA
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const bool lane_predicate = cute::elect_one_sync();

    constexpr int kTmaTransactionBytes = sizeof(ArrayEngine<Element, size(SmemLayout{})>);

    // NOTE: load global memory data to shared memory with tma
    if (warp_idx == 0 && lane_predicate)
    {
        // prefetch TMA descriptors for load
        prefetch_tma_descriptor(tma_load.get_tma_descriptor());

        // get CTA view of gmem tensor
        Tensor gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_layout));
        
        Tensor gmem_tensor_coord_cta = local_tile(gmem_tensor_coord, tile_shape, make_coord(blockIdx.x, blockIdx.y));

        auto tma_load_per_cta = tma_load.get_slice(Int<0>{});

        mbarrier.init(1 /* arrive count */);
        mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);

        cute::copy(tma_load.with(reinterpret_cast<BarrierType& >(mbarrier)),
                   tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
                   tma_load_per_cta.partition_D(smem_tensor));
    }
    __syncthreads();

    mbarrier.wait(0 /* phase */);

    cutlass::arch::fence_view_async_shared();

    // NOTE: store shared memory data to global memory with tma
    if (warp_idx == 0 && lane_predicate)
    {
        // prefetch TMA descriptors for store
        prefetch_tma_descriptor(tma_store.get_tma_descriptor());

        // get CTA view of gmem tensor
        Tensor gmem_tensor_coord = tma_store.get_tma_tensor(shape(gmem_layout));
        
        Tensor gmem_tensor_coord_cta = local_tile(gmem_tensor_coord, tile_shape, make_coord(blockIdx.x, blockIdx.y));

        auto tma_store_per_cta = tma_store.get_slice(Int<0>{});

        cute::copy(tma_store,
                   tma_store_per_cta.partition_S(smem_tensor),
                   tma_store_per_cta.partition_D(gmem_tensor_coord_cta));
    }
}

template <int TileShape_M = 128,
          int TileShape_N = 128,
          int Threads = 32>
int tma_load_and_store(int M, int N, int iterations = 1)
{
    using namespace cute;

    printf("copy with TMA load and store -- no swizzling. \n");

    using Element = float;
    
    auto tensor_shape = make_shape(M, N);

    // allocate and initialize
    thrust::host_vector<Element> host_S(size(tensor_shape)); // (M, N)
    thrust::host_vector<Element> host_D(size(tensor_shape)); // (M, N)

    for (size_t i = 0; i < host_S.size(); ++i)
        host_S[i] = static_cast<Element>(float(i));

    thrust::host_vector<Element> device_S = host_S;
    thrust::host_vector<Element> device_D = host_D;

    // make tensors
    auto gmem_layout_S = make_layout(tensor_shape, LayoutRight{});
    auto gmem_layout_D = make_layout(tensor_shape, LayoutRight{});

    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(device_S.data())), gmem_layout_S);
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(device_D.data())), gmem_layout_D);

    auto tile_shape = make_shape(Int<TileShape_M>{}, Int<TileShape_N>{});

    // NOTE: same smem layout for TMA load and store
    auto smem_layout = make_layout(tile_shape, LayoutRight{});

    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smem_layout);
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smem_layout);

    Params params(tma_load, tma_store, gmem_layout_S, smem_layout, tile_shape);

    dim3 grid_dim(ceil_div(M, TileShape_M), ceil_div(N, TileShape_N));
    dim3 block_dim(Threads);
    dim3 cluster_dim(1);

    int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smem_layout)>));
    printf("smem size: %d. \n", smem_size);

    void const* kernel = (void const *)tma_load_store_kernel<Threads, Element, decltype(params)>;
    set_smem_size(smem_size, kernel);

    // Define the cluster launch paramter structure
    cutlass::ClusterLaunchParams launch_params{grid_dim, block_dim, cluster_dim, smem_size};

    for (int i = 0; i < iterations; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        cutlass::Status status = cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
        cudaError result = cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        if (result != cudaSuccess) 
        {
            std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                        << std::endl;
            return -1;
        }
        std::chrono::duration<double, std::milli> time_diff = t2 - t1;
        double time_ms = time_diff.count();
        std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
                << 2e-6 * M * N * sizeof(Element) / time_ms << " GB/s)"
                << std::endl;
    }

    //
    // Verify
    //

    host_D = device_D;

    int good = 0, bad = 0;

    for (size_t i = 0; i < host_D.size(); ++i)
    {
        if (host_D[i] == host_S[i])
            good++;
        else
            bad++;
    }

    std::cout << "Success " << good << ", Fail " << bad << std::endl;

    return 0;
}

}