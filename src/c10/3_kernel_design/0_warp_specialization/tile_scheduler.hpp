#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor.hpp>
#include <cute/container/tuple.hpp>
#include <cute/util/print.hpp>
#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/fast_math.h>
#include <cutlass/gemm_coord.h>

namespace c1010
{

class SingleTileScheduler
{
public:

    // Host size kernel arguments
    struct Arguments
    {
        int const num_blocks_m, num_block_n, num_blocks_k;
        void* workspace = nullptr;
    };

    // Device side kernel params
    struct Params
    {
        int mn_blocks;
        int num_blocks_k;
    }

    Params const params;

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_blocks_n, args.num_blocks_k};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_blocks_n)};
    }

    static size_t get_workspace_size(int, int) { return 0; }
    static cudaError_t initialize_workspace(int, int, void*, cudaStream_t) { return cudaSuccess; }

    struct WorkTileInfo {
        int M_idx = 0;
        int N_idx = 0;
        bool valid = false;

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, N_idx, 0, params.num_blocks_k};
        }

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const { return valid; }

        CUTLASS_DEVICE
        bool constexpr
        compute_epilogue(Params const& params) const { return true; }

        CUTLASS_DEVICE
        bool constexpr
        needs_fixup(Params const& params) const { return false; }

    };

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), true};
    }

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(WorkTileInfo const& current_work) const {
        return {-1, -1, false};
    }

    CUTLASS_DEVICE
    SingleTileScheduler(Params const& params_) : params(params_) {}


    template <class AccumTensor>
    CUTLASS_DEVICE
    void
    fixup(WorkTileInfo const& worktile,
          AccumTensor& accumulators,
          uint32_t barrier_idx,
          uint32_t thread_idx) const {}
};

}