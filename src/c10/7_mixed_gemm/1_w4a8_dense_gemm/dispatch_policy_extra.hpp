
#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

using namespace cute;

// TODO(Alan): 选择选择的cooperative，后续考虑换成pingpong和FastFP8Accum
struct KernelTmaWarpSpecializedCooperativeX
{
    static constexpr int SchedulerPipelineStageCount = 0;
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic
// schedule With GMMA's A data from registers.
template<int Stages_,
         class ClusterShape_  = Shape<_1, _1, _1>,
         class KernelSchedule = KernelTmaWarpSpecializedCooperativeX>
struct MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInputX
{
    constexpr static int Stages = Stages_;

    using ClusterShape = ClusterShape_;
    using ArchTag      = arch::Sm90;
    using Schedule     = KernelSchedule;

    static_assert(cute::is_same_v<Schedule, KernelTmaWarpSpecialized> ||
                      cute::is_same_v<Schedule, KernelTmaWarpSpecializedPingpong> ||
                      cute::is_same_v<Schedule, KernelTmaWarpSpecializedCooperativeX>,
                  "KernelSchedule must be one of the warp specialized policies");
};


}   // namespace cutlass::gemm

namespace cutlass::epilogue {

struct TmaWarpSpecializedCooperativeX
{

};


//////////////////////////////////////////////////////////////////////////////
//
// Collective Dispatch Policies
//
//////////////////////////////////////////////////////////////////////////////

template<int StagesC_, int StagesD_, int FragmentSize_, bool ReuseSmemC_, bool DelayTmaStore_>
struct Sm90TmaWarpSpecializedX
{
    constexpr static int  StagesC       = StagesC_;
    constexpr static int  StagesD       = StagesD_;
    constexpr static int  FragmentSize  = FragmentSize_;
    constexpr static bool ReuseSmemC    = ReuseSmemC_;
    constexpr static bool DelayTmaStore = DelayTmaStore_;
};

}   // namespace cutlass::epilogue