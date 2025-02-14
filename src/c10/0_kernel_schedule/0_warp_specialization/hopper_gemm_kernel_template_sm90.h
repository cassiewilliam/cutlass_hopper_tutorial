
#pragma once

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include <cute/tensor.hpp>
#include <cutlass/conv/convolution.h>
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include <cutlass/util/packed_stride.hpp>

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

namespace c1000
{

struct HopperFP8GEMM
{
    // A matrix configuration
    using ElementA = cutlass::float_e4m3_t;            // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;         // Layout type for A matrix operand
    static constexpr int AlignmentA 
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A
                                                       // matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = cutlass::float_e4m3_t;            // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;      // Layout type for B matrix operand
    static constexpr int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B
                                                       // matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using ElementC = cutlass::float_e4m3_t;            // Element type for C matrix operands
    using LayoutC = cutlass::layout::RowMajor;         // Layout type for C/D matrix operands (must same with A matrix layout)
    static constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of C/D
                                                       // matrix in units of elements (up to 16 bytes)

    // Output matrix configuration
    using ElementO = cutlass::float_e4m3_t;            // Element type for output matrix operands
    using LayoutO = cutlass::layout::RowMajor;         // Layout type for output matrix operands
    static constexpr int AlignmentOutput 
        = 128 / cutlass::sizeof_bits<ElementO>::value;

    // Multiply-accumulate blocking/pipelining details
    using ElementAccumulator = float;                  // Element type for internal accumulation
    using ElementCompute = float;                      // Element type for compute
    using ArchTag = cutlass::arch::Sm90;               // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass 
        = cutlass::arch::OpClassTensorOp;              // Operator class tag
    
    constexpr int ktile 
        = 128 / cutlass::sizeof_bits<ElementB>::value;
    using _KTile = Int<ktile>;
    using CTAShape = Shape<_128, _128, _KTile>;        // ThreadBlock-Level tile size, 可以通过warp up进行选择
    using TileShape = CTAShape;

    using ClusterShape = Shape<_1, _2, _1>;            // Shape of the threadblocks in a cluster, tuning？

    // Schedule details
    // Mainloop Schedule 如何选择？
    using MainloopScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
        cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
    using KernelSchedule = MainloopScheduleType;

    // Epilogue Schedule 如何选择？
    using EpilogueScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
        cutlass::epilogue::TmaWarpSpecialized,
        cutlass::epilogue::TmaWarpSpecializedCooperative>;
    using EpilogueSchedule = EpilogueScheduleType;

    // Tile Schedule 如何选择？
    using TileSchedulerType = void;
    using TileScheduler = TileSchedulerType;

    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using Activation = cutlass::epilogue::thread::SiLu;

    /* ========================== Collective API ========================== */
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto,
        Activation>::CollectiveOp;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, 
        ClusterShape,
        EpilogueTileType,
        ElementAccumulator,
        ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementO, LayoutO, AlignmentO,
        EpilogueSchedule>::CollectiveOp;

    /* ========================== Kernel API ========================== */
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>, // Indicates ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue,
        TileScheduler>;

    /* ========================== Device API ========================== */
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}