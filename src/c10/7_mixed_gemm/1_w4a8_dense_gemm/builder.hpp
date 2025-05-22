#pragma once

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "dispatch_policy_extra.hpp"
#include "mainloop_sm90_tma_gmma_ws_x.hpp"

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
/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_RS
template<class ElementA_,
         class GmemLayoutATag_,
         int AlignmentA,
         class ElementB_,
         class GmemLayoutBTag_,
         int AlignmentB,
         class ElementAccumulator,
         class TileShape_MNK,
         class ClusterShape_MNK,
         class StageCountType,
         class KernelScheduleType>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA_,
    GmemLayoutATag_,
    AlignmentA,
    ElementB_,
    GmemLayoutBTag_,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
        (cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeX>) &&
        (detail::is_use_rmem_A<ElementA_, GmemLayoutATag_, ElementB_, GmemLayoutBTag_>() ||
         // ConvertAndScale and ConvertAndScaleWithZero
         cute::is_tuple<ElementA_>::value || cute::is_tuple<ElementB_>::value ||
         // DirectConvert
         sizeof_bits<ElementA_>::value != sizeof_bits<ElementB_>::value)>>
{

private:
    using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementA_>;

public:
    // ElementA is uint4b_t
    // ElementB is float_e4m3_t
    using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementA_>;
    using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementB_>;

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

    using GmemLayoutATag = decltype(get_stride(GmemLayoutATag_{}));
    using GmemLayoutBTag = decltype(get_stride(GmemLayoutBTag_{}));

    using ElementPairA = ElementA_;
    using ElementPairB = ElementB_;

    using ElementScale = ScaleA;

    static_assert(is_static<TileShape_MNK>::value);
    static_assert(is_static<ClusterShape_MNK>::value);
    static_assert(
        detail::
            is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
        "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
    static_assert(cutlass::detail::dependent_false<ElementA>,
                  "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
    static constexpr cute::GMMA::Major
        GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
    static constexpr cute::GMMA::Major
        GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
    // If A is scaled, then we don't need to swap. Otherwise, we must ensure B goes to rmem and we
    // must swap the operands.
    static constexpr bool IsWarpSpecializedTransposeB = false;

    // When we relax the above assertion, we must handle setting the tile mma GmmaMajorB correctly.
    static constexpr cute::GMMA::Major TiledMmaGmmaMajorB = GmmaMajorB;

    // For fp32 types, map to tf32 MMA value type.
    using ElementAMma = ElementA;
    using ElementBMma = ElementB;

    // Handle mixed dtypes and MMA.
    using RealElementA    = ElementA;
    using RealElementB    = ElementB;

    using RealElementAMma = ElementB;
    // Always the same for element B.
    using RealElementBMma = ElementB;

    static_assert(TiledMmaGmmaMajorB == GMMA::Major::K || sizeof_bits<ElementB>::value == 16,
                  "Mixed input GEMM does not support MN major layout except for 16bit");

    using AtomLayoutMNK = cute::conditional_t<
        cute::is_any_of_v<KernelScheduleType,
                          KernelTmaWarpSpecializedCooperativeX,
                          KernelPtrArrayTmaWarpSpecializedCooperative>,
        Layout<Shape<_2, _1, _1>>,
        Layout<Shape<_1, _1, _1>>>;


    // ElementB is float8_e4m3_t
    // ElementAccumulator is float
    using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<RealElementAMma,
                                                                              RealElementBMma,
                                                                              ElementAccumulator,
                                                                              TileShape_MNK,
                                                                              GMMA::Major::K,
                                                                              GMMA::Major::K>(),
                                                   AtomLayoutMNK{}));

    using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(
        shape<1>(ClusterShape_MNK{})));

    using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(
        shape<0>(ClusterShape_MNK{})));

    using SmemLayoutAtomA = decltype(detail::rs_smem_selector<
                                     GmmaMajorA,
                                     ElementAMma,
                                     decltype(cute::get<0>(TileShape_MNK{})),
                                     decltype(cute::get<2>(TileShape_MNK{})),
                                     IsWarpSpecializedTransposeB>());
    using SmemLayoutAtomB = decltype(detail::rs_smem_selector<
                                     GmmaMajorB,
                                     ElementBMma,
                                     decltype(cute::get<1>(TileShape_MNK{})),
                                     decltype(cute::get<2>(TileShape_MNK{})),
                                     IsWarpSpecializedTransposeB>());

    static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(
        SmemLayoutAtomA{});
    static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(
        SmemLayoutAtomB{});
    static constexpr int SmemAlignment = static_cast<int>(
        cute::max(SmemAlignmentA, SmemAlignmentB));

    // Handle mixed dtype array GEMM's size of tensor map storage.
    static constexpr size_t TensorMapStorage = sizeof(cute::TmaDescriptor) * 4;

    static constexpr size_t
        SchedulerPipelineStorage = cute::is_pointer_v<TagToStrideA_t<GmemLayoutATag_>>
                                       ? sizeof(
                                             cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>)
                                       : 0;

    static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage +
                                                               SchedulerPipelineStorage);

    static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes -
                                                        KernelSmemCarveout;

    static constexpr int PipelineStages = detail::
        compute_stage_count_or_override_single_affine_transformed_input_no_zero<
            detail::sm90_smem_capacity_bytes,
            RealElementA,
            RealElementB,
            ElementScale,
            TileShape_MNK,
            StageCountType::bytes,
            SmemAlignment>(StageCountType{});

    using SmemCopyAtomA = Copy_Atom<cute::AutoVectorizingCopy, ElementA>;
    using SmemCopyAtomB = void;

    // We pack the scale data with the operand that will be optionally scaled and converted before
    // MMA.
    using StrideA = cute::conditional_t<
        cute::is_layout<cute::remove_pointer_t<GmemLayoutATag_>>::value,
        GmemLayoutATag_,
        TagToStrideA_t<GmemLayoutATag>>;
    using StrideB = cute::conditional_t<
        cute::is_layout<cute::remove_pointer_t<GmemLayoutBTag_>>::value,
        GmemLayoutBTag_,
        TagToStrideB_t<GmemLayoutBTag>>;

    using CollectiveOp = CollectiveMainloop<PipelineStages,
                                            ClusterShape_MNK,
                                            KernelScheduleType,
                                            TileShape_MNK,
                                            ElementPairA,
                                            StrideA,
                                            ElementPairB,
                                            StrideB,
                                            TiledMma,
                                            GmemTiledCopyA,
                                            SmemLayoutAtomA,
                                            SmemCopyAtomA,
                                            cute::identity,
                                            GmemTiledCopyB,
                                            SmemLayoutAtomB,
                                            SmemCopyAtomB,
                                            cute::identity>;

    static_assert(SmemAlignment == static_cast<int>(cute::max(CollectiveOp::SmemAlignmentA,
                                                              CollectiveOp::SmemAlignmentB)));
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}   // namespace cutlass::gemm::collective


namespace cutlass::epilogue::collective {

namespace detail {

// Returns the parameterized dispatch policy for the TMA epilogue
template<class TileShapeMNK, class EpilogueTileMN, class ElementC, class ElementD, class Schedule>
constexpr auto sm90_get_tma_dispatch_policy_x()
{
    using namespace cute;

    constexpr int EpiTiles     = size(shape_div(take<0, 2>(TileShapeMNK{}), EpilogueTileMN{}));
    constexpr int FragmentSize = size(EpilogueTileMN{}) /
                                 (detail::sm90_is_cooperative_v<Schedule> ? 256 : 128);
    // 8b residuals load fast and consume little smem, so the perf cost of waiting on stores to
    // finish outweighs the cost of extra allocation
    constexpr bool ReuseSmem = (sizeof_bits_v<ElementC> == sizeof_bits_v<ElementD>) &&
                               (sizeof_bits_v<ElementD> > 8);
    // TMA store delay performs worse with residual loads and compilicates tensormap updates for
    // Ptr-Array GEMMs
    constexpr bool DelayTmaStore = is_void_v<ElementC> &&
                                   !detail::sm90_is_ptr_array_tma_v<Schedule>;
    constexpr int StagesD = cute::min(EpiTiles, 2);
    constexpr int StagesC = ReuseSmem ? cute::max(cute::min(EpiTiles, 4), StagesD + 1)
                                      : cute::min(EpiTiles, 4);


    return Sm90TmaWarpSpecializedX<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>{};
}

/////////////////////////////////////////////////////////////////////////////////////////////////


// Helper for building TMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template<class TileShape_MNK,
         class EpilogueTile_MN,
         class ElementAccumulator,
         class ElementCompute,
         class ElementC_,
         class GmemLayoutTagC_,
         int AlignmentC,
         class ElementD_,
         class GmemLayoutTagD,
         int AlignmentD,
         class FusionOpOrCallbacks,
         class DispatchPolicy>
struct Sm90TmaBuilderImplX
{
    // Passing void D disables destination store + smem allocation
    using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                                         fusion::get_element_aux_t<FusionOpOrCallbacks>,
                                         ElementD_>;

    // Passing void C disables source load + smem allocation
    using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
                                         ElementD,
                                         ElementC_>;   // prevents void ref breakages

    using GmemLayoutTagC = cute::
        conditional_t<cute::is_void_v<ElementC_>, GmemLayoutTagD, GmemLayoutTagC_>;

    using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
    using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

    using UnderlyingGmemStrideTypeC = cute::remove_pointer_t<GmemStrideTypeC>;
    using UnderlyingGmemStrideTypeD = cute::remove_pointer_t<GmemStrideTypeD>;

    using CopyOpS2G = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagD>,
                                          SM90_TMA_STORE_IM2COL,
                                          SM90_TMA_STORE>;
    using CopyOpG2S = cute::
        conditional_t<detail::is_im2col_mode<GmemLayoutTagC>, SM90_TMA_LOAD_IM2COL, SM90_TMA_LOAD>;

    // Get the smallest tiled copy we can use to retile the accumulators
    using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
    // Get register to register tiled copy that happen before shared memory store.
    // Apply void as no register transform op needed currently.
    using CopyOpR2R = void;

    // TMA builder allows for passing callbacks directly, which is either a fusion::FusionCallbacks
    // instance or a direct visitor implementation, e.g. fusion::Sm90LinearCombination
    using FusionCallbacks = typename CallbacksBuilder<DispatchPolicy,
                                                      FusionOpOrCallbacks,
                                                      TileShape_MNK,
                                                      EpilogueTile_MN,
                                                      ElementAccumulator>::Callbacks;

    using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
        DispatchPolicy,
        TileShape_MNK,
        EpilogueTile_MN,
        ElementC_,   // Need to pass void through to expose via GemmUniversal
        GmemStrideTypeC,
        ElementD_,
        GmemStrideTypeD,
        FusionCallbacks,
        CopyOpG2S,
        decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC,
                                                                    ElementC,
                                                                    EpilogueTile_MN>()),
        decltype(detail::sm90_get_smem_load_op_for_source<UnderlyingGmemStrideTypeC, ElementC>()),
        CopyOpS2G,
        decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD,
                                                                    ElementD,
                                                                    EpilogueTile_MN>()),
        decltype(detail::sm90_get_smem_store_op_for_accumulator<UnderlyingGmemStrideTypeD,
                                                                ElementD>()),
        CopyAtomC,
        CopyOpR2R>;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

}   // namespace detail

// Tma warp-specialized builder
template<class OpClass,
         class TileShape_MNK,
         class ClusterShape_MNK,
         class EpilogueTileType,
         class ElementAccumulator,
         class ElementCompute,
         class ElementC,
         class GmemLayoutTagC,
         int AlignmentC,
         class ElementD_,
         class GmemLayoutTagD,
         int AlignmentD,
         class Schedule,
         class FusionOperation>
struct CollectiveBuilder<
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD_,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    FusionOperation,
    cute::enable_if_t<cute::is_same_v<Schedule, TmaWarpSpecializedCooperativeX> ||
                      detail::sm90_is_ptr_array_tma_v<Schedule>>>
{
private:
    using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                                         fusion::get_element_aux_t<FusionOperation>,
                                         ElementD_>;

    using EpilogueTile_MN = decltype(detail::sm90_compute_tile_shape_or_override<ElementD,
                                                                                 EpilogueTileType,
                                                                                 Schedule,
                                                                                 TileShape_MNK>());

    using DispatchPolicy = decltype(detail::sm90_get_tma_dispatch_policy_x<TileShape_MNK,
                                                                           EpilogueTile_MN,
                                                                           ElementC,
                                                                           ElementD,
                                                                           Schedule>());

public:
    using CollectiveOp = typename detail::Sm90TmaBuilderImplX<TileShape_MNK,
                                                              EpilogueTile_MN,
                                                              ElementAccumulator,
                                                              ElementCompute,
                                                              ElementC,
                                                              GmemLayoutTagC,
                                                              AlignmentC,
                                                              ElementD_,
                                                              GmemLayoutTagD,
                                                              AlignmentD,
                                                              FusionOperation,
                                                              DispatchPolicy>::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}   // namespace cutlass::epilogue::collective
