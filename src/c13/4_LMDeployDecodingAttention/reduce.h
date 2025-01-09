// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

#include "core/array_ops.h"
#include "core/thread_map.h"

#include "cta_map.h"

namespace turbomind::attention {

template<int HeadDim, class T>
void invokeReduce(T*           out,
                  float*       partial_M,
                  float*       partial_L,
                  float*       partial_O,
                  const int*   split_cnt,
                  int          partial_len,
                  int          max_split_cnt,
                  int          query_num,
                  int          head_num,
                  float        exp_scale,
                  cudaStream_t stream);

}  // namespace turbomind::attention
