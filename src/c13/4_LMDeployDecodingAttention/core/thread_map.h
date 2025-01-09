// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "common.h"

#include <iostream>

namespace turbomind {

template<int C, int S, int AccessC, int WarpCount>
struct ThreadMapQ {
    static constexpr int kWarpCount = WarpCount;
    static constexpr int kAccessC   = AccessC;

    static constexpr int kWarpThreadC = C / kAccessC;
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    static_assert(kWarpThreadC <= WARP_SIZE);

    static constexpr int kWarpAccessC = kWarpThreadC * kAccessC;  // C
    static constexpr int kWarpAccessS = kWarpThreadS;

    static constexpr int kWarpIterC = C / kWarpAccessC;  // 1
    static constexpr int kWarpIterS = S / kWarpAccessS;

    static constexpr int kWarpC = 1;
    static constexpr int kWarpS = kWarpCount;

    static constexpr int kIterC = kWarpIterC / kWarpC;  // 1
    static constexpr int kIterS = std::max(kWarpIterS / kWarpS, 1);

    static constexpr int kFootprintC = kWarpAccessC * kIterC;  // C
    static constexpr int kFootprintS = kWarpAccessS * kIterS;

    static constexpr int kDeltaC = kWarpAccessC;
    static constexpr int kDeltaS = kWarpAccessS;

    __device__ static int2 get_offset(int warp_id, int lane_id)
    {
        int warp_offset_c = warp_id % kWarpC;
        int warp_offset_s = warp_id / kWarpC;

        int warp_thread_offset_c = lane_id % kWarpThreadC;
        int warp_thread_offset_s = lane_id / kWarpThreadC;

        int cta_thread_offset_c = kFootprintC * warp_offset_c + warp_thread_offset_c * kAccessC;
        int cta_thread_offset_s = kFootprintS * warp_offset_s + warp_thread_offset_s;

        return {cta_thread_offset_c, cta_thread_offset_s};
    }
};

template<int DimC,                         // head dim 维度 128
         int DimS,                         // sequence len 维度 8
         int AccessC,                      // head_dim维度，每次读取的元素个数
         int WarpCount,                    // 一个Block有几个Warp
         int WarpThreadC = DimC / AccessC> // head_dim维度需要几个线程读取
struct RakedThreadMap {
    static constexpr int kDimC = DimC;
    static constexpr int kDimS = DimS;

    static constexpr int kWarpCount = WarpCount;
    static constexpr int kAccessC   = AccessC;

    static constexpr int kWarpThreadC = WarpThreadC;
    // NOTE: 一个Warp总共有32个线程，首先满足kWarpThreadC，HeadDim维度的读取线程
    //       然后确定SeqLen维度的读取线程
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    static_assert(kWarpThreadC <= WARP_SIZE);

    // NOTE: 一个Warp读取的HeadDim维度元素数量，如果DimC可以被AccessC整除，则DimC一致
    static constexpr int kWarpAccessC = kWarpThreadC * kAccessC;
    // NOTE: 一个Warp读取SeqLen维度的元素数量
    static constexpr int kWarpAccessS = kWarpThreadS;

    // NOTE: 一个Warp需要在HeadDim维度维度迭代几次，如果DimC可以被AccessC整除，则遍历一次即可
    static constexpr int kWarpIterC = (kDimC + kWarpAccessC - 1) / kWarpAccessC;
    // NOTE: 一个Warp需要读取SeqLen维度的几行
    static constexpr int kWarpIterS = kDimS / kWarpAccessS;

    // NOTE: 一个Block有几个Warp读取Head Dim维度的数据
    static constexpr int kWarpC = 1;
    // NOTE: 一个Block有几个Warp用于读取Seq Len维度的数据
    static constexpr int kWarpS = kWarpCount;

    // NOTE: 一个Block需要在HeadDim维度迭代几次
    static constexpr int kIterC = kWarpIterC / kWarpC;
    // NOTE: 一个Block需要在SeqLen维度迭代几次
    static constexpr int kIterS = std::max(kWarpIterS / kWarpS, 1);

    // Allow partial tile when there is ONLY 1 iteration
    static_assert(kDimC % kWarpAccessC == 0 || kIterC == 1);

    static_assert(kIterC > 0);
    static_assert(kIterS > 0);

    // NOTE: 是否为kPartialC，也就是读取的HeadDim维度不是完整的，
    //       当且仅当AccessC不能被DimC整除时发生
    static constexpr bool kPartialC = kDimC % kWarpAccessC != 0;

    // NOTE: 基本等于DimC，也就是一个Block读取的HeadDim维度的数据量
    static constexpr int kFootprintC = kWarpAccessC * kIterC;
    // NOTE: 一个Block读取的SeqLen维度的数据量
    static constexpr int kFootprintS = kWarpAccessS * kIterS;

    static constexpr int kDeltaC = kWarpAccessC;
    static constexpr int kDeltaS = kWarpAccessS;

    // static constexpr int kDeltaC = kWarpAccessC * kWarpC;
    // static constexpr int kDeltaS = kWarpAccessS * kWarpS;

    __device__ static int2 get_offset(int warp_id, int lane_id)
    {
        // NOTE: kWarpC * kWarpS = WarpCount
        //       所以如下代码是确定当前warp_id需要读取
        int block_offset_c = warp_id % kWarpC; // 0
        int block_offset_s = warp_id / kWarpC; // warp_id

        int warp_thread_offset_c = lane_id % kWarpThreadC; // 依据当前Warp的线程，计算需要读取的HeadDim维度的偏移位置
        int warp_thread_offset_s = lane_id / kWarpThreadC; // 依据当前Warp的线程，计算需要读取的SeqLen维度的偏移位置

        int cta_thread_offset_c = kFootprintC * block_offset_c + warp_thread_offset_c * kAccessC;
        int cta_thread_offset_s = kFootprintS * block_offset_s + warp_thread_offset_s;

        // int cta_thread_offset_c = kWarpAccessC * warp_offset_c + warp_thread_offset_c * kAccessC;
        // int cta_thread_offset_s = kWarpAccessS * warp_offset_s + warp_thread_offset_s;

        return {cta_thread_offset_c, cta_thread_offset_s};
    }
};

namespace {

template<class TMap>
void Print(TMap)
{
    std::cout << "     warps: " << TMap::kWarpCount << "\n";
    std::cout << "     shape: (" << TMap::kDimC << ", " << TMap::kDimS << ")\n";
    std::cout << "    access: (" << TMap::kAccessC << ", " << 1 << ")\n";
    std::cout << "warpThread: (" << TMap::kWarpThreadC << ", " << TMap::kWarpThreadS << ")\n";
    std::cout << "warpAccess: (" << TMap::kWarpAccessC << ", " << TMap::kWarpAccessS << ")\n";
    std::cout << "  warpIter: (" << TMap::kWarpIterC << ", " << TMap::kWarpIterS << ")\n";
    std::cout << "      warp: (" << TMap::kWarpC << ", " << TMap::kWarpS << ")\n";
    std::cout << "      iter: (" << TMap::kIterC << ", " << TMap::kIterS << ")\n";
    std::cout << " footprint: (" << TMap::kFootprintC << ", " << TMap::kFootprintS << ")\n";
    std::cout << "     delta: (" << TMap::kDeltaC << ", " << TMap::kDeltaS << ")\n";
    std::cout << "  partialC: " << TMap::kPartialC << "\n";
}

}  // namespace

}  // namespace turbomind
