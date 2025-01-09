// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "macro.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace turbomind {

class RNG {
public:
    RNG();
    ~RNG();
    void GenerateUInt(uint* out, size_t count);

    template<typename T>
    void GenerateUniform(T* out, size_t count, float scale = 1.f, float shift = 0.f);

    template<typename T>
    void GenerateNormal(T* out, size_t count, float scale = 1.f, float shift = 0.f);

    cudaStream_t stream() const;

    void set_stream(cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
