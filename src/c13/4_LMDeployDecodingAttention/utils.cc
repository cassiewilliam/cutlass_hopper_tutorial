// Copyright (c) OpenMMLab. All rights reserved.

#include "utils.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <tuple>

namespace turbomind {

int GetSplitCount(
    int max_split_cnt, int grid_size, int max_active_ctas, int sm_count, int max_wave_cnt, float alpha, float beta)
{
    // NOTE: grid_size,       当前Attention算子需要的Block数量
    //       sm_count,        当前GPU 所包含的sm数量,cudaDevAttrMultiProcessorCount
    //       max_active_ctas, 用于计算在给定内核配置下，每个 SM（Streaming Multiprocessor）上可以同时运行的最大活动块数（CTA，Cooperative Thread Array）。
    //                        也就是每个SM可以同时执行的Block数，每个Block内部可以有多个Warp，32个线程为一个Warp，一个Block或者CTA最多可以分为8个Warp
    // 
    //       cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //          int* numBlocks,                 // 输出参数，返回每个 SM 的最大活动块数
    //          const void* func,               // 内核函数指针
    //          int blockSize,                  // 每个块的线程数
    //          size_t dynamicSMemSize          // 每个块的动态共享内存大小
    //       );
    //
    //       scale, 表示当前Kernel对应的CTA数量需要调用多少次所有GPU SM才能运行完成
    const float scale = (float)grid_size / (sm_count * max_active_ctas);

    // NOTE: 评估split成本的代价函数，split会增加CTA(Block)数量，也就是scale * s表示多少轮调用GPU SM才能运行完成
    //       输入s， 表示split cnt的数量，waves 就是 轮次
    //       成本cost, std::ceil(scale * s) / s，也就是scale * s是整数时候取的最优解
    //       等价cost, s * grid_size / (sm_count * max_active_ctas) 是整数时候取最优解
    //                即可以分析得到，当 split-k 数量使得 运行GPU轮次约均匀越好，感觉存在调优空间 ？
    auto eval = [&](int s) -> std::tuple<float, float, int> {
        float waves = std::ceil(scale * s);
        float cost  = std::numeric_limits<float>::infinity();
        if (s == 1 || waves <= max_wave_cnt) {
            cost = (alpha / s + beta) * waves;
        }
        return {cost, scale * s, s};
    };

    std::tuple<float, float, int> best{std::numeric_limits<float>::infinity(), 0.f, 0};

    auto print = [](auto& x) {  //
        // printf("%d %f %f\n", std::get<2>(x), std::get<1>(x), std::get<0>(x));
    };

    for (int i = 1; i <= max_split_cnt; ++i) {
        auto res = eval(i);
        if (std::isinf(std::get<0>(res))) {
            break;
        }
        print(res);
        if (res < best) {
            best = res;
        }
    }

    print(best);

    return std::get<int>(best);
}

}  // namespace turbomind
