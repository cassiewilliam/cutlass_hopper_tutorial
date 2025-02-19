#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

namespace c1000
{

// writeout RMEM -> GMEM
template <int   kStages,
          class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorageAB
{
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;

    struct {
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline;
    }
};

// writeout RMEM -> SMEM -> GMEM
template <int   kStages,
          class ElementA,
          class ElementB,
          class ElementC,
          class SmemLayoutA,
          class SmemLayoutB,
          class SmemLayoutC>
struct SharedStorageABC
{
    array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
    union {
        array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;
        array_aligned<ElementC, cosize_v<SmemLayoutC>> smem_C;
    };

    struct {
        cutlass::arch::ClusterBarrier barrier_C;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline;
    };
};

template<int      kBlockM_,
         int      kBlockN_,
         int      kBlockK_,
         int      kWarps_,
         int      kStages_,
         int      kClusterM_ = 1,
         int      kClusterN_ = 1,
         typename Element_ = cutlass::half_t,
         typename OutputType_ = Element_,
         bool     FP32Accum = true>
struct KernelTraits
{
    using Element = Element_;
    
    using ElementAccum = std::conditional_t<FP32Accum, float, Element>;
    using OutputType = OutputType_;
    using index_t = int64_t;

    // The Number of threads.
    static constexpr int kWarps = kWarps_;
    static constexpr int kNumThreads = kWarps * cutlass::NumThreadsPerWarp;

    // NOTE: use one warp in producer warpgroup for TMA
    static constexpr int kNumProducerThreads = cutlass::NumThreadsPerWarpGroup;

    static constexpr int kNumConsumerThreads = kNumThreads - kNumProducerThreads;
    static constexpr int kNumWarpGroupMMA = (kWarps / 4) - 1;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<BlockK>>;

    // TODO: Cluster 如何影响Kernel运行？
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kClusterN = kClusterN_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, Int<kClusterN>, _1>

    static constexpr int kStages = kStages_;

    // TODO: What is meaning?
    using AtomLayout_MNK = Layout<Shape<Int<kNumWarpGroupMMA>, _1, _1>

    // TODO: AtomLayoutMNK 决定了什么？
    using TiledMMA = decltype(
        cute::make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
            AtomLayoutMNK{}
        )
    );

    using SmemLayoutAtomA = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::K,
            Element,
            decltype(cute::get<0>(TileShape_MNK{})), // M
            decltype(cute::get<2>(TileShape_MNK{}))
        >()  // K
    );

    using SmemLayoutA = decltype(
        tile_to_shape(
            SmemLayoutAtomA{},
            make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})
        )
    );

    using SmemLayoutAtomB = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::K, 
            Element,
            decltype(cute::get<1>(TileShape_MNK{})),
            decltype(cute::get<2>(TileShape_MNK{}))
        >()
    );

    using SmemLayoutB = decltype(
        tile_to_shape(
            SmemLayoutAtomA{},
            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})
        )
    );

    using SmemLayoutAtomC = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::K,
            OutputType,
            decltype(cute::get<0>(TileShape_MNK{})),
            decltype(cute::get<1>(TileShape_MNK{}))
        >()
    );

    using SmemLayoutC = decltype(
        tile_to_shape(
            SmemLayoutAtomC{},
            select<0, 1>(TileShape_MNK{})
        )
    );

    using SmemCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, OutputType>;

    using SharedStorage = SharedStorageABC<
        kStages, 
        Element,
        Element,
        OutputType,
        SmemLayoutA,
        SmemLayoutB,
        SmemLayoutC
    >;

    // TODO: PipelineTmaAsync是如何实现
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineStage<kStages>;
    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
};

}