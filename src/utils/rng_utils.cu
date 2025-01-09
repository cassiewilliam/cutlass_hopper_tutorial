// Copyright (c) OpenMMLab. All rights reserved.

#include "utils/rng_utils.h"

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

namespace turbomind {

namespace cg = cooperative_groups;

__global__ void curand_init(curandState* state)
{
    auto tid = cg::this_grid().thread_rank();
    curand_init(0xe4c45822e90461ddULL, tid, 0, state + tid);
}

template<typename T>
__global__ void curand_uniform(curandState* state, size_t count, T* result, float scale, float shift)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        float tmp = curand_uniform(state + grid.thread_rank());
        result[i] = T(scale * tmp + shift);
    }
}

template<typename T>
__global__ void curand_normal(curandState* state, size_t count, T* result, float scale, float shift)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        float tmp = curand_normal(state + grid.thread_rank());
        result[i] = T(scale * tmp + shift);
    }
}

__global__ void curand_bytes(curandState* state, size_t count, uint* result)
{
    auto grid = cg::this_grid();
    for (auto i = grid.thread_rank(); i < count; i += grid.size()) {
        result[i] = curand(state + grid.thread_rank());
    }
}

struct RNG::Impl {

    curandState* states{};

    Impl()
    {
        cudaMalloc(&states, sizeof(curandState) * 64 * 64);
        curand_init<<<64, 64>>>(states);
    }

    ~Impl()
    {
        cudaFree(states);
    }

    void GenerateUInt(uint* out, size_t count)
    {
        curand_bytes<<<64, 64, 0, stream_>>>(states, count, out);
    }

    template<typename T>
    void GenerateUniform(T* out, size_t count, float scale, float shift)
    {
        curand_uniform<<<64, 64, 0, stream_>>>(states, count, out, scale, shift);
    }

    template<typename T>
    void GenerateNormal(T* out, size_t count, float scale, float shift)
    {
        curand_normal<<<64, 64, 0, stream_>>>(states, count, out, scale, shift);
    }

    cudaStream_t stream_{};
};

RNG::RNG(): impl_(std::make_unique<Impl>()) {}

RNG::~RNG() = default;

void RNG::GenerateUInt(uint* out, size_t count)
{
    impl_->GenerateUInt(out, count);
}

template<typename T>
void RNG::GenerateUniform(T* out, size_t count, float scale, float shift)
{
    impl_->GenerateUniform(out, count, scale, shift);
}

template<typename T>
void RNG::GenerateNormal(T* out, size_t count, float scale, float shift)
{
    impl_->GenerateNormal(out, count, scale, shift);
}

cudaStream_t RNG::stream() const
{
    return impl_->stream_;
}

void RNG::set_stream(cudaStream_t stream)
{
    impl_->stream_ = stream;
}

template void RNG::GenerateUniform(half* out, size_t count, float scale, float shift);
template void RNG::GenerateUniform(float* out, size_t count, float scale, float shift);
#if ENABLE_BF16
template void RNG::GenerateUniform(nv_bfloat16* out, size_t count, float scale, float shift);
#endif

template void RNG::GenerateNormal(half* out, size_t count, float scale, float shift);
template void RNG::GenerateNormal(float* out, size_t count, float scale, float shift);
#if ENABLE_BF16
template void RNG::GenerateNormal(nv_bfloat16* out, size_t count, float scale, float shift);
#endif

}  // namespace turbomind
