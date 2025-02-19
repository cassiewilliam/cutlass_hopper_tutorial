#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cutlass/gemm/collective/collective_builder.hpp>

namespace c1000
{

template<typename KernelTraits>
struct CollectiveMainloop
{

    using Element = typename KernelTraits::Element;
    using TileShape_MNK = typename KernelTraits::TileShape_MNK;
    using ClusterShape_MNK = typename KernelTraits::ClusterShape_MNK;
    using BarrierType = typename KernelTraits::BarrierType;

    static constexpr int kStages = KernelTraits::BarrierType;

    using SmemLayoutA = typename KernelTraits::SmemLayoutA;
    using SmemLayoutB = typename KernelTraits::SmemLayoutB;

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
    )
    

};

}

