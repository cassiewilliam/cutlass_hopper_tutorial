/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/tensor_predicate.hpp"

#include "dispatch_policy_extra.hpp"
#include "mixed_input_utils_x.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
template<typename KernelTraits>
struct CollectiveMainloop
{
private:
    template<class T>
    friend struct detail::MixedInputUtilsX;
    using CollectiveType = CollectiveMainloop<KernelTraits>;

    using Utils = detail::MixedInputUtilsX<CollectiveType>;

public:
    // Dispatch Policy
    // Stages, ClusterShape, KernelSchedule_
    static constexpr int Stages = KernelTraits::PipelineStages;

    using ArchTag               = typename KernelTraits::ArchTag;
    using ClusterShape          = typename KernelTraits::ClusterShape_MNK;
    using TileShape             = typename KernelTraits::TileShape_MNK;
    using KernelSchedule        = typename KernelTraits::KernelSchedule;
    using ElementAOptionalTuple = typename KernelTraits::ElementAOptionalTuple;
    using StrideA               = typename KernelTraits::StrideA;
    using ElementB              = typename KernelTraits::ElementB;
    using StrideB               = typename KernelTraits::StrideB;
    using TiledMma              = typename KernelTraits::TiledMma;
    using GmemTiledCopyA        = typename KernelTraits::GmemTiledCopyA;
    using GmemTiledCopyB        = typename KernelTraits::GmemTiledCopyB;
    using SmemLayoutAtomA       = typename KernelTraits::SmemLayoutAtomA;
    using SmemLayoutAtomB       = typename KernelTraits::SmemLayoutAtomB;
    using SmemCopyAtomA         = typename KernelTraits::SmemCopyAtomA;
    using SmemCopyAtomB         = typename KernelTraits::SmemCopyAtomB;
    using SmemLayoutScale       = typename KernelTraits::SmemLayoutScale;
    using ScaleTileShape        = typename KernelTraits::ScaleTileShape;

    using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
    using ScaleA   = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;

    using ElementScale = ScaleA;

    // These are always MN major
    using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    // For cases where we can't have a void scale, we can use this to allow the code to compile when
    // the scale is void.
    using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>,
                                                   cute::Stride<_1, int64_t, int64_t>,
                                                   StrideScale>;

    using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));

    using ElementAccumulator = typename TiledMma::ValTypeC;

    using GmemTiledCopyScale = cute::SM90_TMA_LOAD;

    using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopy, ElementScale>;

    // We must ensure the type to be scaled goes to RF
    // ElementA is uint4b_t, UintElementAForBytes to uint4_t
    // ElementB is float8_e4m3_t, UintElementBForBytes uint8_t
    using UintElementAForBytes = uint_bit_t<sizeof_bits_v<ElementA>>;
    using UintElementBForBytes = uint_bit_t<sizeof_bits_v<ElementB>>;

    static constexpr int IsSubbyteA = cute::sizeof_bits_v<UintElementAForBytes> < 8;

    using TmaElementA = uint8_t;
    // in case we have array. translating to uint to satisfy tma descriptor's specialization
    using TmaElementScale = uint_bit_t<sizeof_bits_v<ElementScale>>;

    using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
    using PipelineState    = cutlass::PipelineState<Stages>;

    using PipelineParams = typename MainloopPipeline::Params;

    // One threads per CTA are producers (1 for operand tile)
    static constexpr int NumProducerThreadEvents = 1;

    // Tile along modes in a way that maximizes the TMA box size.

    using SmemLayoutA = decltype(detail::get_smem_layout<Stages>(SmemLayoutAtomA{},
                                                                 select<0, 2>(TileShape{}),
                                                                 StrideA{}));
    using SmemLayoutB = decltype(detail::get_smem_layout<Stages>(SmemLayoutAtomB{},
                                                                 select<1, 2>(TileShape{}),
                                                                 StrideB{}));

public:
    static constexpr bool UseScaleLookupTable = cutlass::detail::is_Array_v<ElementScale>;

    static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{});

    static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});

    // Just pick the max alignment of A and B since it is required to be at least 128B
    static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);

    static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128,
                  "Require at least 128B alignment");

    struct SharedStorage
    {
        static constexpr int scale_elements = Utils::elements_per_smem_scale();
        struct TensorStorage
        {
            CUTE_ALIGNAS(SmemAlignmentA)
            cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
            CUTE_ALIGNAS(SmemAlignmentB)
            cute::ArrayEngine<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
            cute::ArrayEngine<ElementScale, scale_elements>                             smem_scale;
        } tensors;

        using PipelineStorage = typename MainloopPipeline::SharedStorage;
        PipelineStorage pipeline;
    };
    using TensorStorage   = typename SharedStorage::TensorStorage;
    using PipelineStorage = typename SharedStorage::PipelineStorage;

    // Host side kernel arguments
    struct Arguments
    {
        ElementA const*     ptr_A = nullptr;
        StrideA             dA{};
        ElementB const*     ptr_B = nullptr;
        StrideB             dB{};
        ElementScale const* ptr_S = nullptr;
        NonVoidStrideScale  dS{};
        int                 group_size             = 0;
        uint32_t            mma_promotion_interval = 4;
    };

    // Device side kernel params
    struct Params
    {
    public:
        // Assumption: StrideA is congruent with Problem_MK
        using LayoutA = decltype(detail::get_gmem_layout(repeat_like(StrideA{}, int32_t(0)),
                                                         StrideA{}));
        using LayoutB = decltype(detail::get_gmem_layout(repeat_like(StrideB{}, int32_t(0)),
                                                         StrideB{}));

        using TMA_A = decltype(make_tma_copy_A_sm90<TmaElementA>(
            GmemTiledCopyA{},
            make_tensor(detail::get_logical_ptr(static_cast<UintElementAForBytes const*>(nullptr)),
                        LayoutA{}),
            SmemLayoutA{}(_, _, cute::Int<0>{}),
            TileShape{},
            ClusterShape{}));   // mcast along N mode for this M load, if any

        using TMA_Scale = decltype(make_tma_copy<TmaElementScale>(
            GmemTiledCopyScale{},
            make_tensor(detail::get_logical_ptr(static_cast<ElementScale const*>(nullptr)),
                        repeat_like(NonVoidStrideScale{}, int32_t(0)),
                        NonVoidStrideScale{}),
            SmemLayoutScale{}(_, _, cute::Int<0>{}),
            ScaleTileShape{},
            _1{}));   // mcast along N mode for this M load, if any. Scale is ALWAYS loaded with A
                      // for RF kernel

        // Assumption: StrideB is congruent with Problem_NK
        using TMA_B = decltype(make_tma_copy_B_sm90(
            GmemTiledCopyB{},
            make_tensor(detail::get_logical_ptr(static_cast<UintElementBForBytes const*>(nullptr)),
                        LayoutB{}),
            SmemLayoutB{}(_, _, cute::Int<0>{}),
            TileShape{},
            ClusterShape{}));   // mcast along M mode for this N load, if any
        TMA_A     tma_load_a;
        TMA_B     tma_load_b;
        TMA_Scale tma_load_scale;
        int64_t   scale_k;
        int       group_size;
        uint32_t  tma_transaction_bytes = TmaTransactionBytes;
        int       reload_factor = (group_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
        StrideA   dA;
        StrideB   dB;
    };

    //
    // Methods
    //

    template<class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                    Arguments const&    args,
                                                    void*               workspace)
    {
        (void)workspace;

        // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3
        // (MNK)
        auto problem_shape_MNKL = append<4>(problem_shape, 1);
        auto [M, N, K, L]       = problem_shape_MNKL;

        UintElementAForBytes const* ptr_A;
        StrideA                     dA;
        UintElementBForBytes const* ptr_B;
        StrideB                     dB;

        ptr_A = reinterpret_cast<UintElementAForBytes const*>(args.ptr_A);
        ptr_B = reinterpret_cast<UintElementBForBytes const*>(args.ptr_B);
        dA    = args.dA;
        dB    = args.dB;

        Tensor tensor_a = make_tensor(detail::get_logical_ptr(ptr_A),
                                      detail::get_gmem_layout(make_shape(M, K, L), dA));
        Tensor tensor_b = make_tensor(detail::get_logical_ptr(ptr_B),
                                      detail::get_gmem_layout(make_shape(N, K, L), dB));

        typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90<TmaElementA>(
            GmemTiledCopyA{},
            tensor_a,
            SmemLayoutA{}(_, _, cute::Int<0>{}),
            TileShape{},
            ClusterShape{});   // mcast along N mode for this M load, if any

        typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
            GmemTiledCopyB{},
            tensor_b,
            SmemLayoutB{}(_, _, cute::Int<0>{}),
            TileShape{},
            ClusterShape{});   // mcast along M mode for this N load, if any

        typename Params::TMA_Scale tma_load_scale{};

        uint32_t tma_transaction_bytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

        // NOTE(Alan): has scales
        {
            auto                scale_k      = ceil_div(K, args.group_size);
            ElementScale const* ptr_S        = args.ptr_S;
            StrideScale         dS           = args.dS;
            Tensor              tensor_scale = make_tensor(detail::get_logical_ptr(ptr_S),
                                              make_layout(make_shape(M, scale_k, L), dS));
            tma_load_scale                   = make_tma_copy<TmaElementScale>(
                GmemTiledCopyScale{},
                tensor_scale,
                SmemLayoutScale{}(_, _, cute::Int<0>{}),
                ScaleTileShape{},
                _1{});   // mcast along N mode for this M load, if any

            return {tma_load_a,
                    tma_load_b,
                    tma_load_scale,
                    scale_k,
                    args.group_size,
                    tma_transaction_bytes + TmaTransactionBytesExtra,
                    (args.group_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}),
                    dA,
                    dB};
        }
    }

    template<class ProblemShape>
    static bool can_implement(ProblemShape const&               problem_shape,
                              [[maybe_unused]] Arguments const& args)
    {
        constexpr int tma_alignment_bits = 128;
        auto          problem_shape_MNKL = append<4>(problem_shape, 1);
        auto [M, N, K, L]                = problem_shape_MNKL;

        constexpr int min_tma_aligned_elements_A = tma_alignment_bits /
                                                   cutlass::sizeof_bits<ElementA>::value;

        bool check_aligned_A = cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
            detail::get_gmem_layout(cute::make_shape(M, K, L), args.dA));

        constexpr int min_tma_aligned_elements_B = tma_alignment_bits /
                                                   cutlass::sizeof_bits<ElementB>::value;

        bool check_aligned_B = cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
            detail::get_gmem_layout(cute::make_shape(N, K, L), args.dB));

        bool check_aligned_S = true;
        bool check_aligned_Z = true;
        bool check_mode_args = true;

        // NOTE(Alan): has scales
        {
            const int scale_mn = M;
            const int scale_k  = ceil_div(K, args.group_size);
            constexpr int
                min_tma_aligned_elements_scale = tma_alignment_bits /
                                                 cutlass::sizeof_bits<ElementScale>::value;
            check_aligned_S = cutlass::detail::check_alignment<min_tma_aligned_elements_scale>(
                cute::make_shape(scale_mn, scale_k, L),
                args.dS);
            check_mode_args = check_mode_args && (args.group_size == K ||
                                                  ((args.group_size % size<2>(TileShape{})) == 0));
            check_mode_args = check_mode_args && args.group_size != 0;
            check_mode_args = check_mode_args && (args.ptr_S != nullptr);
        }

        if (!check_mode_args)
        {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Invalid arguments for the selected conversion mode.\n");
        }
        if (!check_aligned_A)
        {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Tensor A meet the minimum alignment requirements for TMA.\n");
        }
        if (!check_aligned_B)
        {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Tensor B meet the minimum alignment requirements for TMA.\n");
        }
        if (!check_aligned_S)
        {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor S (scale) meet the minimum alignment "
                               "requirements for TMA.\n");
        }
        if (!check_aligned_Z)
        {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Tensor Z (zeros) meet the minimum alignment "
                               "requirements for TMA.\n");
        }

        return check_mode_args && check_aligned_A && check_aligned_B && check_aligned_S &&
               check_aligned_Z;
    }

    static constexpr int      K_PIPE_MAX            = Stages;
    static constexpr uint32_t TmaTransactionBytesMK = Utils::compute_tma_transaction_bytes_mk();
    static constexpr uint32_t TmaTransactionBytesNK = Utils::compute_tma_transaction_bytes_nk();
    static constexpr uint32_t
        TmaTransactionBytesExtra                  = Utils::compute_tma_transaction_bytes_extra();
    static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK +
                                                    TmaTransactionBytesExtra;

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params)
    {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());

        cute::prefetch_tma_descriptor(mainloop_params.tma_load_scale.get_tma_descriptor());
    }

    /// Set up the data needed by this collective for load and mma.
    /// Returns a tuple of tensors. The collective and the kernel layer have the contract
    /// Returned tuple must contain at least two elements, with the first two elements being:
    /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
    /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
    /// The rest of the tensors can be specified as needed by this collective.
    template<class ProblemShape_MNKL>
    CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
                                  Params const&            mainloop_params) const
    {
        using X = Underscore;
        // Separate out problem shape for convenience
        auto [M, N, K, L] = problem_shape_MNKL;

        // TMA requires special handling of strides to deal with coord codomain mapping
        // Represent the full tensors -- get these from TMA
        Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(
            shape(detail::get_gmem_layout(make_shape(M, K, L), mainloop_params.dA)));   // (m,k,l)
        Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
            shape(detail::get_gmem_layout(make_shape(N, K, L), mainloop_params.dB)));   // (n,k,l)

        // Make tiled views, defer the slice
        Tensor gA_mkl = local_tile(mA_mkl,
                                   TileShape{},
                                   make_coord(_, _, _),
                                   Step<_1, X, _1>{});   // (BLK_M,BLK_K,m,k,l)
        Tensor gB_nkl = local_tile(mB_nkl,
                                   TileShape{},
                                   make_coord(_, _, _),
                                   Step<X, _1, _1>{});   // (BLK_N,BLK_K,n,k,l)

        // else if constexpr (ModeHasScales)
        {
            auto   scale_k = mainloop_params.scale_k;
            Tensor mS_mkl  = mainloop_params.tma_load_scale.get_tma_tensor(
                make_shape(M, scale_k, L));   // (m,scale_k,l)
            Tensor gS_mkl = local_tile(mS_mkl,
                                       ScaleTileShape{},
                                       make_coord(_, _));   // (BLK_M,BLK_Scale_K,m,scale_k,l)

            return cute::make_tuple(gA_mkl, gB_nkl, gS_mkl);
        }
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Producer Perspective
    /// This overload gets triggered when we have scales.
    template<class... Ts, class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void load(Params const&             mainloop_params,
                             MainloopPipeline          pipeline,
                             PipelineState             smem_pipe_write,
                             cute::tuple<Ts...> const& load_inputs,
                             BlockCoord const&         blk_coord,
                             KTileIterator             k_tile_iter,
                             int                       k_tile_count,
                             int                       thread_idx,
                             uint32_t                  block_rank_in_cluster,
                             TensorStorage&            shared_tensors)
    {
        static_assert(sizeof...(Ts) == 3, "Scaled convert needs three inputs");

        Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                                 SmemLayoutA{});   // (BLK_M,BLK_K,PIPE)
        Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                                 SmemLayoutB{});                    // (BLK_N,BLK_K,PIPE)
        Tensor sA  = as_position_independent_swizzle_tensor(sA_);   // (BLK_M,BLK_K,PIPE)
        Tensor sB  = as_position_independent_swizzle_tensor(sB_);   // (BLK_N,BLK_K,PIPE)

        //
        // Prepare the TMA loads for A, B and Scales
        //

        constexpr uint32_t cluster_shape_x        = get<0>(ClusterShape());
        uint2              cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                                     block_rank_in_cluster / cluster_shape_x};

        Tensor gA_mkl = get<0>(load_inputs);
        Tensor gB_nkl = get<1>(load_inputs);

        auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
        auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

        // Partition the inputs based on the current block coordinates.
        auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
        Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);   // (BLK_M,BLK_K,k)
        Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);   // (BLK_N,BLK_K,k)

        // Applies the mapping from block_tma_a
        Tensor tAgA = block_tma_a.partition_S(gA);   // (TMA,TMA_M,TMA_K,k)
        Tensor tAsA = block_tma_a.partition_D(sA);   // (TMA,TMA_M,TMA_K,PIPE)

        Tensor tBgB = block_tma_b.partition_S(gB);   // (TMA,TMA_N,TMA_K,k)
        Tensor tBsB = block_tma_b.partition_D(sB);   // (TMA,TMA_N,TMA_K,PIPE)

        uint16_t mcast_mask_a = 0;
        uint16_t mcast_mask_b = 0;
        uint16_t mcast_mask_s = 0;

        // Issue TmaLoads
        // Maps the tile -> block, value
        if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>)
        {
            auto block_layout = Layout<ClusterShape>{};   // (m,n) ->
                                                          // block_id
            for (int n = 0; n < size<1>(block_layout); ++n)
            {
                mcast_mask_a |= (uint16_t(1)
                                 << block_layout(cluster_local_block_id.x, n, Int<0>{}));
            }
        }

        if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>)
        {
            auto block_layout = Layout<ClusterShape>{};   // (m,n) ->
                                                          // block_id
            for (int m = 0; m < size<0>(block_layout); ++m)
            {
                mcast_mask_b |= (uint16_t(1)
                                 << block_layout(m, cluster_local_block_id.y, Int<0>{}));
            }
        }

        auto extra_input_partitions = Utils::partition_extra_tma_inputs(mainloop_params,
                                                                        load_inputs,
                                                                        shared_tensors,
                                                                        cluster_local_block_id,
                                                                        m_coord,
                                                                        l_coord);

        // Mainloop
        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count)
        {
            // LOCK smem_pipe_write for _writing_
            pipeline.producer_acquire(smem_pipe_write);

            //
            // Copy gmem to smem for *k_tile_iter
            //

            using BarrierType        = typename MainloopPipeline::ProducerBarrierType;
            BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

            int write_stage = smem_pipe_write.index();
            if (cute::elect_one_sync())
            {
                copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a),
                     tAgA(_, _, _, *k_tile_iter),
                     tAsA(_, _, _, write_stage));
                copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b),
                     tBgB(_, _, _, *k_tile_iter),
                     tBsB(_, _, _, write_stage));
            }

            // else if constexpr (ModeHasScales)
            {
                auto tSgS = get<0>(extra_input_partitions);
                auto tSsS = get<1>(extra_input_partitions);

                // Temporary factor which will determine which k tile to reload from gmem. Needed so
                // we don't modify tma transaction bytes on the fly. We must do a ceiling divide
                // here to correctly handle with group_size == K. In that case, we don't require
                // that K is a multiple of the threadblock tile K
                int const scale_load_k = *k_tile_iter /
                                         mainloop_params.reload_factor;   // This will always be 0
                                                                          // when group_size == K.
                if (cute::elect_one_sync())
                    copy(mainloop_params.tma_load_scale.with(*tma_barrier, mcast_mask_s),
                         tSgS(_, _, _, scale_load_k),
                         tSsS(_, _, _, write_stage));
            }

            ++k_tile_iter;

            // Advance smem_pipe_write
            ++smem_pipe_write;
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write)
    {
        // Issue the epilogue waits
        if (cute::elect_one_sync())
        {
            /* This helps avoid early exit of blocks in Cluster
             * Waits for all stages to either be released (all
             * Consumer UNLOCKs), or if the stage was never used
             * then would just be acquired since the phase was
             * still inverted from make_producer_start_state
             */
            pipeline.producer_tail(smem_pipe_write);
        }
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Consumer Perspective
    template<class FrgTensorC>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline,
                            PipelineState    smem_pipe_read,
                            FrgTensorC&      accum,
                            int              k_tile_count,
                            int              thread_idx,
                            TensorStorage&   shared_tensors,
                            Params const&    mainloop_params)
    {
        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtomA must be rank 2.");
        static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtomB must be rank 2.");
        static_assert(
            !cute::is_void_v<SmemCopyAtomA>,
            "SM90 GMMA mainloops must specify a non-void copy atom for RF sourced instructions.");
        static_assert(
            cute::is_void_v<SmemCopyAtomB>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

        // Obtain warp index
        int                  warp_idx              = canonical_warp_idx_sync();
        [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

        Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                                 SmemLayoutA{});                    // (BLK_M,BLK_K,PIPE)
        Tensor sA  = as_position_independent_swizzle_tensor(sA_);   // (BLK_M,BLK_K,PIPE)

        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                                SmemLayoutB{});   // (BLK_N,BLK_K,PIPE)

        //
        // Define C accumulators and A/B partitioning
        //

        // Layout of warp group to thread mapping

        static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                          size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be "
                      "NumThreadsPerWarpGroup");

        constexpr int MmaWarpGroups            = size(TiledMma{}) / NumThreadsPerWarpGroup;
        Layout        warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                      Int<NumThreadsPerWarpGroup>{});

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

        TiledMma tiled_mma;
        auto     mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
        Tensor   tCsA             = mma_thread_slice.partition_A(sA);
        auto mma_warpgroup_slice  = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

        // Allocate fragments and descriptors
        Tensor tCrA_mma = mma_thread_slice.partition_fragment_A(
            sA(_, _, Int<0>{}));   // (MMA,MMA_M,MMA_K,PIPE)

        Tensor tCrA_load = [&] {
            if constexpr (not is_layout<StrideA>::value)
            {
                // Make register tensor with MMA layout
                return make_fragment_like<ElementA>(tCrA_mma);
            }
            else
            {
                // Make register tensor matching smem layout, converter will take care of
                // de-swizzling
                return make_tensor_like<ElementA>(tCsA(_, _, _, Int<0>{}));
            }
        }();

        Tensor tCsB = mma_warpgroup_slice.partition_B(sB);         // (MMA,MMA_N,MMA_K,PIPE)
        Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);   // (MMA,MMA_N,MMA_K,PIPE)

        //
        // Copy Atom A retiling
        //
        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA_load);   // (CPY,CPY_M,CPY_K)

        // Partition of thread -> shared and thread -> RF
        auto partitioned_extra_info     = Utils::partition_extra_mma_info(mma_thread_slice,
                                                                      shared_tensors);
        auto copy_partitions_extra_info = Utils::retile_extra_mma_info(tiled_mma,
                                                                       partitioned_extra_info,
                                                                       warp_group_thread_idx);

        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));   // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));   // CPY_K
        CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));        // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));            // N
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));             // K
        CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));             // PIPE
        CUTE_STATIC_ASSERT_V(Int<Stages>{} == size<2>(sA));               // PIPE
        CUTE_STATIC_ASSERT_V(Int<Stages>{} == size<2>(sB));               // PIPE

        //
        // PIPELINED MAIN LOOP
        //

        // We release buffers to producer warps(dma load) with some mmas in flight
        PipelineState smem_pipe_release = smem_pipe_read;

        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(accum);

        constexpr int K_BLOCK_MAX = size<2>(tCrA_load);
        constexpr int K_WAIT_MAX  = cute::min(K_BLOCK_MAX - 1, 7);
        static_assert(K_BLOCK_MAX >= 4, "Consider increasing TileShapeK");

        ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
        // first k tile
        {
            barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            int read_stage = smem_pipe_read.index();

            ++smem_pipe_read;
            barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

            // copy smem->rmem for A operand
            Utils::copy_tensors_MK(smem_tiled_copy_A,
                                   tCsA,
                                   tCrA_copy_view,
                                   partitioned_extra_info,
                                   copy_partitions_extra_info,
                                   0,
                                   read_stage);
            if (K_BLOCK_MAX > 1)
            {   // prefetch next block
                Utils::copy_tensors_MK(smem_tiled_copy_A,
                                       tCsA,
                                       tCrA_copy_view,
                                       partitioned_extra_info,
                                       copy_partitions_extra_info,
                                       1,
                                       read_stage);
            }
            Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);

            // Unroll the K mode manually to set scale D to 1
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
            {
                warpgroup_arrive();
                // (V,M) x (V,N) => (V,M,N)
                cute::gemm(tiled_mma,
                           tCrA_mma(_, _, k_block),
                           tCrB(_, _, k_block, read_stage),
                           accum);
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                warpgroup_commit_batch();

                if (k_block < K_BLOCK_MAX - 2)
                {   // prefetch next block
                    Utils::copy_tensors_MK(smem_tiled_copy_A,
                                           tCsA,
                                           tCrA_copy_view,
                                           partitioned_extra_info,
                                           copy_partitions_extra_info,
                                           k_block + 2,
                                           read_stage);
                }
                if (k_block < K_BLOCK_MAX - 1)
                {
                    Utils::dequantize_A_kblock(tCrA_load,
                                               tCrA_mma,
                                               partitioned_extra_info,
                                               k_block + 1);
                }
            }

            --k_tile_count;
            if (k_tile_count > 0)
            {
                // Wait for K_BLOCK_MAX - 1 to be in flight to ensure that it is safe to overwrite
                // the A registers for the first mma.
                pipeline.consumer_wait(smem_pipe_read, barrier_token);
                Utils::copy_tensors_MK(smem_tiled_copy_A,
                                       tCsA,
                                       tCrA_copy_view,
                                       partitioned_extra_info,
                                       copy_partitions_extra_info,
                                       0,
                                       smem_pipe_read.index());
                if (K_BLOCK_MAX > 1)
                {   // prefetch next block
                    Utils::copy_tensors_MK(smem_tiled_copy_A,
                                           tCsA,
                                           tCrA_copy_view,
                                           partitioned_extra_info,
                                           copy_partitions_extra_info,
                                           1,
                                           smem_pipe_read.index());
                }
                warpgroup_wait<K_WAIT_MAX>();
                Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);
            }
        }

        if (k_tile_count == 0)
        {
            return;
        }

        warpgroup_fence_operand(accum);
        // Mainloop GMMAs
        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 1; --k_tile_count)
        {

            //
            // Compute on k_tile
            //

            int read_stage = smem_pipe_read.index();
            ++smem_pipe_read;

            warpgroup_fence_operand(accum);
            // Unroll the K mode manually to set scale D to 1
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
            {

                warpgroup_arrive();
                // (V,M) x (V,N) => (V,M,N)
                cute::gemm(tiled_mma,
                           tCrA_mma(_, _, k_block),
                           tCrB(_, _, k_block, read_stage),
                           accum);
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                warpgroup_commit_batch();

                warpgroup_wait<K_WAIT_MAX>();   // We have K_BLOCK_MAX - 1 GMMA instructions pending
                                                // for this stage, so we can release prior barrier
                if (k_block == K_BLOCK_MAX - 1)
                {
                    pipeline.consumer_release(
                        smem_pipe_release);   // UNLOCK smem_pipe_release, done _computing_ on it
                    ++smem_pipe_release;
                }

                if (k_block == 0)
                {
                    barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
                }

                if (k_block == K_BLOCK_MAX - 1)
                {
                    pipeline.consumer_wait(smem_pipe_read, barrier_token);
                    Utils::copy_tensors_MK(smem_tiled_copy_A,
                                           tCsA,
                                           tCrA_copy_view,
                                           partitioned_extra_info,
                                           copy_partitions_extra_info,
                                           0,
                                           smem_pipe_read.index());
                    if (K_BLOCK_MAX > 1)
                    {   // prefetch next block
                        Utils::copy_tensors_MK(smem_tiled_copy_A,
                                               tCsA,
                                               tCrA_copy_view,
                                               partitioned_extra_info,
                                               copy_partitions_extra_info,
                                               1,
                                               smem_pipe_read.index());
                    }
                    Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);
                }
                else
                {
                    if (k_block < K_BLOCK_MAX - 2)
                    {   // prefetch next block
                        Utils::copy_tensors_MK(smem_tiled_copy_A,
                                               tCsA,
                                               tCrA_copy_view,
                                               partitioned_extra_info,
                                               copy_partitions_extra_info,
                                               k_block + 2,
                                               read_stage);
                    }
                    Utils::dequantize_A_kblock(tCrA_load,
                                               tCrA_mma,
                                               partitioned_extra_info,
                                               k_block + 1);
                }
            }
            warpgroup_fence_operand(accum);
        }

        warpgroup_fence_operand(accum);

        {
            //
            // Compute on k_tile
            //

            int read_stage = smem_pipe_read.index();

            warpgroup_fence_operand(accum);

            // Unroll the K mode manually to set scale D to 1
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
            {

                warpgroup_arrive();
                // (V,M) x (V,N) => (V,M,N)
                cute::gemm(tiled_mma,
                           tCrA_mma(_, _, k_block),
                           tCrB(_, _, k_block, read_stage),
                           accum);
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                warpgroup_commit_batch();

                warpgroup_wait<K_WAIT_MAX>();
                if (k_block == K_BLOCK_MAX - 1)
                {   // release prior barrier
                    pipeline.consumer_release(
                        smem_pipe_release);   // UNLOCK smem_pipe_release, done _computing_ on it
                    ++smem_pipe_release;
                }

                if (k_block < K_BLOCK_MAX - 2)
                {   // prefetch next block
                    Utils::copy_tensors_MK(smem_tiled_copy_A,
                                           tCsA,
                                           tCrA_copy_view,
                                           partitioned_extra_info,
                                           copy_partitions_extra_info,
                                           k_block + 2,
                                           read_stage);
                }
                if (k_block < K_BLOCK_MAX - 1)
                {
                    Utils::dequantize_A_kblock(tCrA_load,
                                               tCrA_mma,
                                               partitioned_extra_info,
                                               k_block + 1);
                }
            }
        }

        warpgroup_fence_operand(accum);
    }

    /// Perform a Consumer Epilogue to release all buffers
    CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline,
                                 PipelineState    smem_pipe_release,
                                 int              k_tile_count)
    {
        // Prologue GMMAs
        int prologue_mma_count = 1;
        k_tile_count -= prologue_mma_count;

        smem_pipe_release.advance(k_tile_count);

        // Wait on all GMMAs to complete
        warpgroup_wait<0>();

        for (int count = 0; count < prologue_mma_count; ++count)
        {
            pipeline.consumer_release(
                smem_pipe_release);   // UNLOCK smem_pipe_release, done _computing_ on it
            ++smem_pipe_release;
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}   // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
