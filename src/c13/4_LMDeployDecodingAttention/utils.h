// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

enum QuantPolicy
{
    kNone = 0x00,
    // reserve 0x01 and 0x02 for backward compatibility
    kReserve1 = 0x01,
    kReserve2 = 0x02,
    // quantize cache kv
    kCacheKVInt8 = 0x08, // 8
    kCacheKVInt4 = 0x04, // 4
};

int GetSplitCount(int   max_split_cnt,
                  int   grid_size,
                  int   max_active_ctas,
                  int   sm_count,
                  int   max_wave_cnt,
                  float alpha = 1,
                  float beta  = 1e-3);

}
