
#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

using namespace cute;

// TODO(Alan): 选择选择的cooperative，后续考虑换成pingpong和FastFP8Accum
struct KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaling {};

// Mixed precision version n-buffer in rmem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule for Ptr-Array and Grouped Gemm
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaling
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInputBlockScaling {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaling> ||
    cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedPingpong>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative policies");
};

}