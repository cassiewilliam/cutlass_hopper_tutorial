#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cutlass/gemm/collective/collective_builder.hpp>

namespace c1010
{

template<typename KTraits>
struct CollectiveMainloop
{

    using Element = typename KTraits::Element;
    using TileShape_MNK = typename KTraits::TileShape_MNK;
    using ClusterShape_MNK = typename KTraits::ClusterShape_MNK;
    using BarrierType = typename KTraits::BarrierType;

    static constexpr int kStages = KTraits::kStages;

    using SmemLayoutA = typename KTraits::SmemLayoutA;
    using SmemLayoutB = typename KTraits::SmemLayoutB;

    using ShapeT = cute::Shape<int32_t, int32_t>;
    using StrideT = cute::Shape<int32_t, _1>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using TMA_A = decltype(
        make_tma_copy(
            cute::SM90_TMA_LOAD{},
            make_tensor(
                make_gmem_ptr(static_cast<Element const*>(nullptr)),
                ShapeT{},
                StrideT{}
            ),
            take<0, 2>(SmemLayoutA{}),
            select<0, 2>(TileShape_MNK{}),
            _1{}
        )
    );

    using TMA_B = decltype(
        make_tma_copy(
            cute::SM90_TMA_LOAD{},
            make_tensor(
                make_gmem_ptr(static_cast<Element const*>(nullptr)),
                ShapeT{},
                StrideT{}
            ),
            take<0, 2>(SmemLayoutA{}),
            select<1, 2>(TileShape_MNK{}),
            _1{}
        )
    );

    static constexpr int kNumMmaThreads = KTraits::kNumMmaThreads;
    using MainloopPipeline = typename KTraits::MainloopPipeline;

    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    // set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t kTmaTransactionBytesA = 
        static_cast<uint32_t>(size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t kTmaTransactionBytesB = 
        static_cast<uint32_t>(size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<Element> / 8);
    
    static constexpr uint32_t kTmaTransactionBytes = kTmaTransactionBytesA + kTmaTransactionBytesB;

    // Host side kernel arguments
    struct Arguments
    {
        Element const* ptr_A;
        LayoutT layout_A;
        Element const* ptr_B;
        LayoutT layout_B;
    };

    struct Arguments
    {
        LayoutT layout_A;
        LayoutT layout_B;
        TMA_A   tma_load_A;
        TMA_B   tma_load_B;
    };

    static Params
    to_underlying_arguments(Arguments const& args)
    {
        Tensor gmem_A = make_tensor(make_gmem_ptr(args.ptr_A), args.layout_A);
        Tensor gmem_B = make_tensor(make_gmem_ptr(args.ptr_B), args.layout_B);

        TMA_A tma_load_A = make_tma_copy(
            cute::SM90_TMA_LOAD{},
            gmem_A,
            SmemLayoutA{}(_, _, _0{}),
            select<0, 2>(TileShape_MNK{}),
            _1{} // no multi-cast
        );

        TMA_B tma_load_B = make_tma_coppy(

        );
    }

};

}

