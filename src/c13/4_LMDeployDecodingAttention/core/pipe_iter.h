// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

template<int Stages, int Step = 1>
struct PipeIter {
    static constexpr int kMaxStep = Stages * Step;

    int r = 0;
    int w = kMaxStep - Step;

    // NOTE: 前 ++ 运算符重载，先执行递增操作，然后返回递增后的对象
    //       不需要创建临时对象，比较高效
    __inline__ __device__ PipeIter& operator++()
    {
        w = r;
        r += Step;
        if (r == kMaxStep) {
            r -= kMaxStep;
        }
        return *this;
    }
};

}  // namespace turbomind
