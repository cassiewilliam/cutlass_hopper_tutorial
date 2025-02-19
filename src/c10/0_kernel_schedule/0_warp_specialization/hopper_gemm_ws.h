#pragma once

#include <vector>
#include <cuda_runtime_api.h>

namespace c1000
{
/**
 * @brief 用于hopper_gemm_ws运行支持的接口
 */
class HopperGemmWsRunnerInterface
{
public:
    HopperGemmWsRunnerInterface() {}

    virtual HopperGemmWsRunnerInterface() {}

    virtual void gemm(void*        D,
                      void* const* A,
                      void* const* B,
                      void* const* C,
                      int          m,
                      int          n,
                      int          k,
                      float        scale_A,
                      float        scale_B,
                      char*        workspace,
                      size_t       workspace_bytes,
                      cudaStream_t stream,
                      int*         occupancy = nullptr) = 0;
    
    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<CutlassGemmConfig> getConfigs() const = 0;
};

template <typename T>
class HopperGemmWsRunner : public virtual HopperGemmWsRunnerInterface
{
public:
    HopperGemmWsRunner();
    ~HopperGemmWsRunner();


    virtual void gemm(void*        D,
                      void* const* A,
                      void* const* B,
                      void* const* C,
                      int          m,
                      int          n,
                      int          k,
                      float        scale_A,
                      float        scale_B,
                      char*        workspace,
                      size_t       workspace_bytes,
                      cudaStream_t stream,
                      int*         occupancy = nullptr) override;
};

}