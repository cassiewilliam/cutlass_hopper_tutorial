#pragma once

#include <cuda_runtime.h>

#include "compiler.cuh"
#include "jit_tunner.cuh"

namespace c108
{

class DeepGemmRunner
{
public:

    DeepGemmRunner()
    {
        m_jittuner = std::make_shared<deep_gemm::jit::JITTuner>();
    }

    ~DeepGemmRunner() {}

    void tunning(int m, int n, int k, deep_gemm::GemmType type, cudaStream_t stream = 0);

    void gemm(int          m,
              int          n,
              int          k,
              void*        A,
              void*        A_scales,
              void*        B,
              void*        B_scales,
              void*        C,
              cudaStream_t stream = 0);

    void group_gemm_contiguous(int          m,
                                int          n,
                                int          k,
                                float        alpha,
                                void const*  A,
                                int          lda,
                                void const*  B,
                                int          ldb,
                                float        beta,
                                void*        C,
                                int          ldc,
                                cudaStream_t stream = 0);

    void group_gemm_masked(int          m,
                            int          n,
                            int          k,
                            float        alpha,
                            void const*  A,
                            int          lda,
                            void const*  B,
                            int          ldb,
                            float        beta,
                            void*        C,
                            int          ldc,
                            cudaStream_t stream = 0);

private:

    std::shared_ptr<deep_gemm::jit::JITTuner> m_jittuner;
};

}