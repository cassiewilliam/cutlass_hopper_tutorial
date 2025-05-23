#pragma once

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "dispatch_policy_extra.hpp"
#include "mainloop_sm90_tma_gmma_ws_x.hpp"

namespace cutlass::epilogue::fusion {

// D = alpha * acc
template<int  StagesC,
         int  StagesD,
         int  FragmentSize,
         bool ReuseSmemC,
         bool DelayTmaStore,
         class ElementOutput,
         class ElementCompute,
         class ElementScalar,
         FloatRoundStyle RoundStyle,
         class CtaTileShapeMNK,
         class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecializedX<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile> : Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
                            Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
                            Sm90AccFetch>
{
    using Impl = Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
                         Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,
                         Sm90AccFetch>;

    using Operation = fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>;

    struct Arguments
    {
        // Give a name and flat ordering to the fusion callback args
        ElementScalar        alpha     = ElementScalar(1);
        ElementScalar        beta      = ElementScalar(0);
        ElementScalar const* alpha_ptr = nullptr;
        ElementScalar const* beta_ptr  = nullptr;

        using StrideAlpha  = Stride<_0, _0, int64_t>;
        StrideAlpha dAlpha = {_0{}, _0{}, 0};

        // Conversion to the args expected by the visitor implementation
        // to_underlying_arguments will implicitly call this
        operator typename Impl::Arguments() const
        {
            return {
                // binary op : alpha * acc
                {{alpha}, {alpha_ptr}, {dAlpha}},   // leaf args : alpha
                {},                                 // leaf args : acc
                {}                                  // binary args : multiplies
            };   // end binary op
        }
    };

    // Ctor inheritance
    using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////


}   // namespace cutlass::epilogue::fusion


namespace cutlass::epilogue::collective {

namespace detail {


// Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to
// provide one.
template<class ElementD, class EpilogueTileType, class Schedule, class TileShape_MNK>
constexpr auto sm90_compute_tile_shape_or_override_x()
{
    if constexpr (cute::is_same_v<EpilogueTileType, EpilogueTileAuto>)
    {
        auto epi_tile = [&]() {
            if constexpr (cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecializedCooperativeX,
                                             Schedule>)
            {
                auto tile_m = cute::min(_128{}, size<0>(TileShape_MNK{}));
                auto tile_n = cute::gcd(cute::min(_32{}, size<1>(TileShape_MNK{})),
                                        size<1>(TileShape_MNK{}));
                return make_shape(tile_m, tile_n);
            }
            else if constexpr (detail::sm90_is_warp_specialized_v<Schedule>)
            {
                constexpr int N_perf = sizeof_bits_v<ElementD> == 8 ? 64 : 32;
                auto          tile_m = cute::min(_64{}, size<0>(TileShape_MNK{}));
                auto          tile_n = cute::gcd(cute::min(Int<N_perf>{}, size<1>(TileShape_MNK{})),
                                        size<1>(TileShape_MNK{}));
                return make_shape(tile_m, tile_n);
            }
            else
            {
                static_assert(cutlass::detail::dependent_false<Schedule>, "Unsupported schedule.");
            }
        }();

        return cute::transform(epi_tile, seq<0, 1>{}, [](auto epi_tiler, auto I) {
            auto cta_tiler = make_layout(get<I>(TileShape_MNK{}));
            // This is a multimodal CTA tiler, transform before returning
            if constexpr (depth(cta_tiler) > 0)
            {
                // This is an implicit multimodal tiler, match profile and return
                if constexpr (tuple_size_v<decltype(shape(cta_tiler))> == 1)
                {
                    return make_tile(epi_tiler);
                }
                // This is an explicit multimodal tiler, compose out epi tiler
                else
                {
                    return composition(cta_tiler, epi_tiler);
                }
            }
            // This is a flat CTA tiler, no need for transformation
            else
            {
                return epi_tiler;
            }
        });
    }
    else if constexpr (cute::is_tuple<EpilogueTileType>::value)
    {
        EpilogueTileType epi_tile;
        constexpr int    M = size<0>(shape(epi_tile));
        constexpr int    N = size<1>(shape(epi_tile));

        static_assert(!is_layout<EpilogueTileType>::value,
                      "EpilogueTile must be a cute::Tile or cute::Shape");
        static_assert(M == 64 && detail::sm90_is_warp_specialized_v<Schedule> ||
                          M == 128 && detail::sm90_is_cooperative_v<Schedule>,
                      "Unsupported tile shape");
        static_assert(N % 16 == 0, "Unsupported tile shape");

        return epi_tile;
    }
    else
    {
        static_assert(cutlass::detail::dependent_false<EpilogueTileType>,
                      "Invalid type for EpilogueTileType.");
    }
}

// Returns the parameterized dispatch policy for the TMA epilogue
template<class TileShapeMNK, class EpilogueTileMN, class ElementC, class ElementD, class Schedule>
constexpr auto sm90_get_tma_dispatch_policy_x()
{
    using namespace cute;

    constexpr int EpiTiles     = size(shape_div(take<0, 2>(TileShapeMNK{}), EpilogueTileMN{}));
    constexpr int
        FragmentSize = size(EpilogueTileMN{}) /
                       (cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecializedCooperativeX,
                                           Schedule>
                            ? 256
                            : 128);
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

    using EpilogueTile_MN = decltype(detail::sm90_compute_tile_shape_or_override_x<
                                     ElementD,
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
