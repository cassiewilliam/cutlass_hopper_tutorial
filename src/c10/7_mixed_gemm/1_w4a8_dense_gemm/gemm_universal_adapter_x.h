#pragma once

#include "cutlass/gemm/device/gemm_universal_adapter.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::device {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace detail {

// 检测类型 T 是否有成员类型 DispatchPolicy
template<typename T, typename = void>
struct has_dispatch_policy : std::false_type
{};

template<typename T>
struct has_dispatch_policy<T, std::void_t<typename T::DispatchPolicy>> : std::true_type
{};

}   // namespace detail

/*!
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::Gemm or cutlass::gemm::kernel::GemmUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behaviour might
  differ between the two specializations.
*/
template<class GemmKernel_, class Enable = void>
class GemmUniversalAdapterX;

template<class GemmKernel_>
class GemmUniversalAdapterX<GemmKernel_,
                            cute::enable_if_t<!detail::has_dispatch_policy<GemmKernel_>::value>>
{
public:
    using GemmKernel         = GetUnderlyingKernel_t<GemmKernel_>;
    using TileShape          = typename GemmKernel::TileShape;
    using ElementA           = typename GemmKernel::ElementA;
    using ElementB           = typename GemmKernel::ElementB;
    using ElementC           = typename GemmKernel::ElementC;
    using ElementD           = typename GemmKernel::ElementD;
    using ElementAccumulator = typename GemmKernel::ElementAccumulator;
    using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
    using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

    // Map back to 2.x type as best as possible
    using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
    using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
    using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
    using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

    // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
    using MathOperator = cutlass::arch::OpMultiplyAdd;

    using OperatorClass = cutlass::detail::get_operator_class_t<
        typename CollectiveMainloop::TiledMma>;

    using ArchTag = typename GemmKernel::ArchTag;

    // NOTE: Assume identity swizzle for now
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
    using ThreadblockShape = cutlass::gemm::GemmShape<cute::size<0>(TileShape{}),
                                                      cute::size<1>(TileShape{}),
                                                      cute::size<2>(TileShape{})>;

    using ClusterShape = cutlass::gemm::GemmShape<
        cute::size<0>(typename GemmKernel::ClusterShape{}),
        cute::size<1>(typename GemmKernel::ClusterShape{}),
        cute::size<2>(typename GemmKernel::ClusterShape{})>;

    // Instruction shape is easy too, since we get that directly from our TiledMma's atom shape
    using InstructionShape = cutlass::gemm::GemmShape<
        cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
        cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
        cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

    // Legacy: provide a correct warp count, but no reliable warp shape
    static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

    // Warp shape is not a primary API type in 3.x
    // But we can best approximate it by inspecting the TiledMma
    // For this, we make the assumption that we always have 4 warps along M, and rest along N, none
    // along K We also always round up the warp count to 4 if the tiled mma is smaller than 128
    // threads
    static constexpr int WarpsInMma = cute::max(
        4,
        CUTE_STATIC_V(cute::size(typename GemmKernel::TiledMma{})) / 32);

    static constexpr int WarpsInMmaM = 4;
    static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);

    using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;

    using WarpShape = cutlass::gemm::GemmShape<
        CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
        CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
        CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

    static int constexpr kStages = CollectiveMainloop::Stages;

    using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

    /// Argument structure: User API
    using Arguments = typename GemmKernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename GemmKernel::Params;

private:
    /// Kernel API parameters object
    Params params_;

public:
    /// Access the Params structure
    Params const& params() const
    {
        return params_;
    }

    /// Determines whether the GEMM can execute the given problem.
    static Status can_implement(Arguments const& args)
    {
        if (GemmKernel::can_implement(args))
        {
            return Status::kSuccess;
        }
        else
        {
            return Status::kInvalid;
        }
    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args)
    {
        size_t workspace_bytes = 0;
        if (args.mode == GemmUniversalMode::kGemmSplitKParallel)
        {
            workspace_bytes += sizeof(int) * size_t(cute::size<0>(TileShape{})) *
                               size_t(cute::size<1>(TileShape{}));
        }

        workspace_bytes += GemmKernel::get_workspace_size(args);

        CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

        return workspace_bytes;
    }

    /// Initializes GEMM state from arguments.
    Status initialize(Arguments const& args,
                      void*            workspace    = nullptr,
                      cudaStream_t     stream       = nullptr,
                      CudaHostAdapter* cuda_adapter = nullptr)
    {

        CUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
                           << workspace << ", stream: " << (stream ? "non-null" : "null"));

        // Initialize the workspace
        Status status = GemmKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
        if (status != Status::kSuccess)
        {
            return status;
        }
        // Initialize the Params structure
        params_ = GemmKernel::to_underlying_arguments(args, workspace);
        // Don't set the function attributes - require the CudaHostAdapter to set it.

        //
        // Account for dynamic smem capacity if needed
        //
        int smem_size = GemmKernel::SharedStorageSize;

        CUTLASS_ASSERT(cuda_adapter == nullptr);

        if (smem_size >= (48 << 10))
        {
            CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
            cudaError_t result = cudaFuncSetAttribute(device_kernel<GemmKernel>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                      smem_size);
            if (cudaSuccess != result)
            {
                result = cudaGetLastError();   // to clear the error bit
                CUTLASS_TRACE_HOST(
                    "  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
                return Status::kErrorInternal;
            }
        }
        return Status::kSuccess;
    }

    /// Primary run() entry point API that is static allowing users to create and manage their own
    /// params. Supplied params struct must be construct by calling
    /// GemmKernel::to_underlying_arguments()
    static Status run(Params&          params,
                      cudaStream_t     stream          = nullptr,
                      CudaHostAdapter* cuda_adapter    = nullptr,
                      bool             launch_with_pdl = false)
    {
        CUTLASS_TRACE_HOST("GemmUniversal::run()");

        dim3 const block = GemmKernel::get_block_shape();
        dim3 const grid  = GemmKernel::get_grid_shape(params);

        dim3 cluster(cute::size<0>(typename GemmKernel::ClusterShape{}),
                     cute::size<1>(typename GemmKernel::ClusterShape{}),
                     cute::size<2>(typename GemmKernel::ClusterShape{}));

        // configure smem size and carveout
        int smem_size = GemmKernel::SharedStorageSize;

        Status launch_result{Status::kSuccess};
        // Use extended launch API only for mainloops that use it

        constexpr bool is_static_1x1x1 = cute::is_static_v<typename GemmKernel::ClusterShape> and
                                         cute::size(typename GemmKernel::ClusterShape{}) == 1;


        [[maybe_unused]] void* kernel_params[] = {&params};
        CUTLASS_ASSERT(cuda_adapter == nullptr);

        [[maybe_unused]] void const* kernel = (void const*)device_kernel<GemmKernel>;

        static constexpr bool kClusterLaunch = GemmKernel::ArchTag::kMinComputeCapability == 90;

        if constexpr (kClusterLaunch)
        {
            if constexpr (is_static_1x1x1)
            {
                launch_result = cutlass::kernel_launch<GemmKernel>(grid,
                                                                   block,
                                                                   smem_size,
                                                                   stream,
                                                                   params,
                                                                   launch_with_pdl);
                if (launch_result != Status::kSuccess)
                {
                    CUTLASS_TRACE_HOST(
                        "GemmUniversal::run: cutlass::kernel_launch reports failure");
                }
            }
            else
            {
                launch_result = ClusterLauncher::launch(grid,
                                                        cluster,
                                                        block,
                                                        smem_size,
                                                        stream,
                                                        kernel,
                                                        kernel_params,
                                                        launch_with_pdl);
            }
        }

        cudaError_t result = cudaGetLastError();
        if (cudaSuccess == result && Status::kSuccess == launch_result)
        {
            return Status::kSuccess;
        }
        else
        {
            CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
            return Status::kErrorInternal;
        }
    }

    /// Overload that allows a user to re-launch the same kernel without updating internal params
    /// struct.
    Status run(cudaStream_t     stream          = nullptr,
               CudaHostAdapter* cuda_adapter    = nullptr,
               bool             launch_with_pdl = false)
    {
        return run(params_, stream, cuda_adapter, launch_with_pdl);
    }
};

////////////////////////////////////////////////////////////////////////////////

}   // namespace cutlass::gemm::device

////////////////////////////////////////////////////////////////////////////////
