// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_fp16.h>
#include <memory>

#include "attention_params.h"
#include "utils/macro.h"

namespace turbomind {

template<typename T>
void Compare(
    const T* src, const T* ref, size_t stride, int m, int n, bool show = false, float rtol = 1e-2, float atol = 1e-4);

void LoadBinary(const std::string& path, size_t size, void* dst);

class RNG {
public:
    RNG();
    ~RNG();
    void GenerateUInt(uint* out, size_t count);

    template<typename T>
    void GenerateUniform(T* out, size_t count, float scale = 1.f, float shift = 0.f);

    template<typename T>
    void GenerateNormal(T* out, size_t count, float scale = 1.f, float shift = 0.f);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

template<typename T>
void mmha_ft_reference(const AttentionParams<T>& params,
                       T**                       per_sample_k_cache,
                       T**                       per_sample_v_cache,
                       const int*                sequence_length,
                       int                       max_memory_len,
                       cudaStream_t              st);

}  // namespace turbomind
