#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace cutlass::gemm::collective {

namespace detail {
// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an
// optional scale matrix), or overrides with manual count.
template<int capacity_bytes_,
         class ElementA,
         class ElementB,
         class ElementScale,
         class TileShapeMNK,
         int carveout_bytes_,
         int alignment = 128>
constexpr int compute_stage_count_or_override_single_affine_transformed_input_no_zero(
    StageCountAutoCarveout<carveout_bytes_> stage_count)
{

    // 32 bytes to account for barriers etc.
    constexpr auto mainloop_pipeline_bytes = sizeof(
        typename cutlass::PipelineTmaAsync<1>::SharedStorage);
    constexpr int scale_zero_k_tile = 1;

    constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
    constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;

    constexpr auto s_bits = get_bits_for_possibly_void_element<ElementScale>();

    constexpr auto scale_bytes = cutlass::bits_to_bytes(s_bits * size<0>(TileShapeMNK{}) *
                                                        scale_zero_k_tile);

    static_assert(scale_bytes % 128 == 0, "Scale bytes must be a multiple of 128");

    // When scales are void, s_bits will be 0 so no smem will be allocated for scales.
    constexpr int stage_bytes_ = cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) *
                                                        size<2>(TileShapeMNK{})) +
                                 cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) *
                                                        size<2>(TileShapeMNK{})) +
                                 scale_bytes;

    constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
                                static_cast<int>(mainloop_pipeline_bytes);

    constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
    constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

    return (capacity_bytes - carveout_bytes) / stage_bytes;
}
}   // namespace detail
}   // namespace cutlass::gemm::collective


namespace cutlass_w4a8 {


template<class T>
static auto get_stride(T const& t)
{
    if constexpr (not cute::is_layout<cute::remove_pointer_t<T>>::value)
    {
        return t;
    }
    else
    {
        if constexpr (cute::is_pointer_v<T>)
        {
            return &cute::stride(*t);
        }
        else
        {
            return cute::stride(t);
        }
    }
}

template<typename OutType, typename CollectiveEpilogue>
struct MixedInputGemmKernelTraits
{
    using MmaType   = cutlass::float_e4m3_t;
    using QuantType = cutlass::int4b_t;

    static constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // GEMM with TN support
    // A matrix configuration
    // NOTE(Alan): A is Weight
    using ElementA = QuantType;                      // Element type for B matrix operand
    using LayoutA  = cutlass::layout::ColumnMajor;   // Layout type for B matrix operand

    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    // B matrix configuration
    // NOTE(Alan): B is Activateion
    using ElementB = MmaType;
    using LayoutB  = cutlass::layout::RowMajor;   // Layout type for A matrix operand

    // Alignment of A matrix in units of elements (up to 16 bytes)
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    // Define the CuTe layout for reoredered quantized tensor B
    // LayoutAtomQuant places values that will be read by the same thread in contiguous locations in
    // global memory. It specifies the reordering within a single warp's fragment
    using LayoutBtomQuant   = decltype(cutlass::compute_memory_reordering_atom<ElementB>());

    using LayoutA_Reordered = decltype(cute::tile_to_shape(
        LayoutBtomQuant{},
        cute::Layout<cute::Shape<int, int, int>, cutlass::detail::TagToStrideA_t<LayoutA>>{}));

    // This example manually swaps and transposes, so keep transpose of input layouts
    using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

    using GmemLayoutATag = decltype(get_stride(LayoutA_Reordered{}));
    using GmemLayoutBTag = decltype(get_stride(LayoutB_Transpose{}));

    // We pack the scale data with the operand that will be optionally scaled and converted before
    // MMA.
    using StrideA = cute::conditional_t<
        cute::is_layout<cute::remove_pointer_t<LayoutA_Reordered>>::value,
        LayoutA_Reordered,
        cutlass::detail::TagToStrideA_t<GmemLayoutATag>>;

    using StrideB = cute::conditional_t<
        cute::is_layout<cute::remove_pointer_t<LayoutB_Transpose>>::value,
        LayoutB_Transpose,
        cutlass::detail::TagToStrideB_t<GmemLayoutBTag>>;

    // TODO(Alan): 考虑换成float16
    using ElementScale = cutlass::float_e4m3_t;
    using LayoutScale  = cutlass::layout::RowMajor;

    using ElementAOptionalTuple = cute::tuple<ElementA, cutlass::Array<ElementScale, 8>>;

    // D matrix configuration
    using ElementD           = cutlass::half_t;
    using LayoutD            = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // For fp32 types, map to tf32 MMA value type.
    using ElementAMma = ElementA;
    using ElementBMma = ElementB;

    // Handle mixed dtypes and MMA.
    using RealElementA = ElementA;
    using RealElementB = ElementB;

    // NOTE(Alan): A输入的时候是int4b_t, 计算的时候是采用float8_e4m3_t
    using RealElementAMma = ElementB;
    // Always the same for element B.
    using RealElementBMma = ElementB;

    // Core kernel configurations
    using ElementAccumulator = float;      // Element type for internal accumulation
    using ArchTag = cutlass::arch::Sm90;   // Tag indicating the minimum SM that supports the
                                           // intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp;   // Operator class tag


    // TODO(Alan): 考虑换成tuning
    using kBlockM = cute::_128;
    using kBlockN = cute::_128;
    using kBlockK = cute::Int<TileShapeK>;

    using TileShape_MNK = cute::Shape<kBlockM, kBlockN, kBlockK>;   // Threadblock-level tile size
    using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;   // Shape of the
                                                                          // threadblocks in a
                                                                          // cluster

    // Kernel to launch based on the default setting in the // Collective Builder
    using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperativeX;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;

    static constexpr cute::GMMA::Major
        GmmaMajorA = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_A<LayoutA_Reordered>();

    static constexpr cute::GMMA::Major
        GmmaMajorB = cutlass::gemm::collective::detail::gmma_rs_tag_to_major_B<LayoutB_Transpose>();

    using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::rs_smem_selector<
                                     GmmaMajorA,
                                     ElementAMma,
                                     decltype(cute::get<0>(TileShape_MNK{})),
                                     decltype(cute::get<2>(TileShape_MNK{})),
                                     false /* IsWarpSpecializedTransposeB */>());

    using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::rs_smem_selector<
                                     GmmaMajorB,
                                     ElementBMma,
                                     decltype(cute::get<1>(TileShape_MNK{})),
                                     decltype(cute::get<2>(TileShape_MNK{})),
                                     false /* IsWarpSpecializedTransposeB */>());

    static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(
        SmemLayoutAtomA{});
    static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(
        SmemLayoutAtomB{});

    static constexpr int SmemAlignment = static_cast<int>(
        cute::max(SmemAlignmentA, SmemAlignmentB));

    static constexpr int PipelineStages = cutlass::gemm::collective::detail::
        compute_stage_count_or_override_single_affine_transformed_input_no_zero<
            cutlass::gemm::collective::detail::sm90_smem_capacity_bytes,
            ElementA,
            ElementB,
            ElementScale,
            TileShape_MNK,
            StageCountType::bytes,
            SmemAlignment>(StageCountType{});


    using SmemLayoutA = decltype(cutlass::gemm::collective::detail::get_smem_layout<PipelineStages>(
        SmemLayoutAtomA{},
        cute::select<0, 2>(TileShape_MNK{}),
        StrideA{}));

    using SmemLayoutB = decltype(cutlass::gemm::collective::detail::get_smem_layout<PipelineStages>(
        SmemLayoutAtomB{},
        cute::select<1, 2>(TileShape_MNK{}),
        StrideB{}));

    using SmemLayoutAtomScale = cute::Layout<
        cute::Shape<decltype(cute::shape<0>(SmemLayoutAtomA{})), cute::Int<1>>>;

    using ScaleTileShape = decltype(make_shape(cute::shape<0>(TileShape_MNK{}),
                                               cute::shape<1>(SmemLayoutAtomScale{})));

    // These are always MN major
    using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;

    // For cases where we can't have a void scale, we can use this to allow the code to compile when
    // the scale is void.
    using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>,
                                                   cute::Stride<cute::_1, int64_t, int64_t>,
                                                   StrideScale>;

    // It is assumed that the scales and zero-points share the same smem layout
    using SmemLayoutScale = decltype(tile_to_shape(
        SmemLayoutAtomScale{},
        make_shape(cute::shape<0>(ScaleTileShape{}),
                   cute::shape<1>(ScaleTileShape{}),
                   cute::Int<PipelineStages>{}),
        cute::conditional_t<::cutlass::gemm::detail::is_major<0, NonVoidStrideScale>(),
                            cute::Step<cute::_2, cute::_1, cute::_3>,
                            cute::Step<cute::_1, cute::_2, cute::_3>>{}));
    using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::
                                        sm90_cluster_shape_to_tma_atom(
                                            cute::shape<1>(ClusterShape_MNK{})));

    using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::
                                        sm90_cluster_shape_to_tma_atom(
                                            cute::shape<0>(ClusterShape_MNK{})));


    using AtomLayoutMNK = cute::conditional_t<
        cute::is_same_v<KernelSchedule, cutlass::gemm::KernelTmaWarpSpecializedCooperativeX>,
        cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>,
        cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>>;

    using SmemCopyAtomA = cute::Copy_Atom<cute::AutoVectorizingCopy, ElementA>;
    using SmemCopyAtomB = void;

    // ElementB is float8_e4m3_t
    // ElementAccumulator is float
    using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<RealElementAMma,
                                                                              RealElementBMma,
                                                                              ElementAccumulator,
                                                                              TileShape_MNK,
                                                                              cute::GMMA::Major::K,
                                                                              cute::GMMA::Major::K>(),
                                                   AtomLayoutMNK{}));
};

}   // namespace cutlass_w4a8