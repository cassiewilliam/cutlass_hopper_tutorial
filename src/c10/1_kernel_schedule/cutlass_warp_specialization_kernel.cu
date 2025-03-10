#pragma once

#include <cuda_runtime.h>
#include <iostream>

#include <cutlass/cutlass.h>

#include <cute/tensor.hpp>
#include <cutlass/tensor_ref.h>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>

#include <cutlass/util/command_line.h>
#include <cutlass/util/distribution.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/reference/device/tensor_fill.h>

namespace c101
{
/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  RasterOrderOptions raster;
  int swizzle;

  Options():
    help(false),
    m(5120), n(4096), k(4096),
    alpha(1.f), beta(0.f),
    iterations(1000),
    raster(RasterOrderOptions::Heuristic),
    swizzle(1)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "48_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct KernelTraits
{
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // A matrix configuration
    using         ElementA    = float;                                          // Element type for A matrix operand
    using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = float;                                          // Element type for B matrix operand
    using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using         ElementC    = float;                                          // Element type for C and D matrix operands
    using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
    constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

    // Core kernel configurations
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
    using TileShape           = Shape<_128,_128,_32>;                           // Threadblock-level tile size
    using ClusterShape        = Shape<_4,_2,_1>;                                // Shape of the threadblocks in a cluster
    using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
    using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementC, LayoutC, AlignmentC,
        cutlass::epilogue::collective::EpilogueScheduleAuto
      >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
      >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int>, // Indicates ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Reference device GEMM implementation type
    using DeviceGemmReference = cutlass::reference::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        ElementAccumulator>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}


template <typename KernelTraits>
struct WarpSpecialGemm
{

    template<class Element>
    bool initialize_block(cutlass::DeviceAllocation<Element>& block,
                          uint64_t                            seed = 2025)
    {
        Element scope_max, scope_min;
        int bits_input = cutlass::sizeof_bits<Element>::value;

        if (bits_input == 1)
        {
          scope_max = Element(2);
          scope_min = Element(0);
        }
        else if (bits_input <= 8)
        {
          scope_max = Element(2);
          scope_min = Element(-2);
        } 
        else
        {
          scope_max = Element(8);
          scope_min = Element(-8);
        }

        cutlass::reference::device::BlockFillRandomUniform(block.get(),
                                                           block.size(),
                                                           seed,
                                                           scope_max,
                                                           scope_min,
                                                           0);
        return true;
    }

    // Initialize operands to be used in the GEMM and reference GEMM
    void initialize(const Options& options)
    {
        stride_A = cutlass::make_cute_packed_stride(KernelTraits::StrideA{}, {options.m, options.k, 1});
        stride_B = cutlass::make_cute_packed_stride(KernelTraits::StrideB{}, {options.n, options.k, 1});
        stride_C = cutlass::make_cute_packed_stride(KernelTraits::StrideC{}, {options.m, options.n, 1});
        stride_D = cutlass::make_cute_packed_stride(KernelTraits::StrideD{}, {options.m, options.n, 1});

        block_A.reset(options.m * options.k);
        block_B.reset(options.k * options.n);
        block_C.reset(options.m * options.n);
        block_D.reset(options.m * options.n);
        block_D_ref.reset(options.m * options.n);

        initialize_block(block_A, seed + 203);
        initialize_block(block_B, seed + 202);
        initialize_block(block_C, seed + 201);
    }

    // Populated a GEMM::Arguments structure from the given commandline options
    typename KernelTraits::Gemm::Arguments to_underlying_arguments(const Options& options)
    {
        // Change device_id to another value if you are running on a machine with multiple GPUs and wish
        // to use a GPU other than that with device ID 0.
        int device_id = 0;
        cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

        typename KernelTraits::Gemm::Arguments arguments
        {
            cutlass::gemm::GemmUniversalMode::kGemm,
            {options.m, options.n, options.k},
            {block_A.get(), stride_A, block_B.get(), stride_B},
            {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
            kernel_hw_info
        };

        arguments.scheduler.raster_order = options.raster;
        arguments.scheduler.max_swizzle_size = options.swizzle;

        return arguments;
    }

    bool verify(const Options& options)
    {
        cutlass::TensorRef ref_A(block_A.get(), KernelTraits::Gemm::LayoutA::packed({options.m, options.k}));
        cutlass::TensorRef ref_B(block_B.get(), KernelTraits::Gemm::LayoutB::packed({options.k, options.n}));
        cutlass::TensorRef ref_C(block_C.get(), KernelTraits::Gemm::LayoutC::packed({options.m, options.n}));
        cutlass::TensorRef ref_D(block_D_ref.get(), KernelTraits::Gemm::LayoutC::packed({options.m, options.n}));

        // Create instantiation for device reference gemm kernel
        DeviceGemmReference gemm_reference;

        // Launch device reference gemm kernel
        gemm_reference(
          {options.m, options.n, options.k},
          ElementAccumulator(options.alpha),
          ref_A,
          ref_B,
          ElementAccumulator(options.beta),
          ref_C,
          ref_D);

        // Wait for kernel to finish
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

        return passed;
    }

    int run(Options& options)
    {
        initialize(options);

        // Instantiate CUTLASS kernel depending on templates
        KernelTraits::Gemm gemm;

        // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
        auto arguments = args_from_options(options);

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = KernelTraits::Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

        // Correctness / Warmup iteration
        CUTLASS_CHECK(gemm.run());

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        Result result;
        result.passed = verify(options);

        std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

        if (!result.passed)
        {
            exit(-1);
        }

        // Run profiling loop
        if (options.iterations > 0)
        {
          GpuTimer timer;
          timer.start();
          for (int iter = 0; iter < options.iterations; ++iter) {
            CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
            CUTLASS_CHECK(gemm.run());
          }
          timer.stop();

          // Compute average runtime and GFLOPs.
          float elapsed_ms = timer.elapsed_millis();
          result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
          result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

          std::string raster = "Heuristic";

          if (options.raster == RasterOrderOptions::AlongN) {
            raster = "Along N";
          }
          else if (options.raster == RasterOrderOptions::AlongM) {
            raster = "Along M";
          }

          std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
          std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
          std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
          std::cout << "  GFLOPS: " << result.gflops << std::endl;
        }

  return 0;

    }

public:
    //
    // Data members
    //

    /// Initialization
    KernelTraits::StrideA stride_A;
    KernelTraits::StrideB stride_B;
    KernelTraits::StrideC stride_C;
    KernelTraits::StrideD stride_D;
    uint64_t seed;

    cutlass::DeviceAllocation<typename KernelTraits::Gemm::ElementA> block_A;
    cutlass::DeviceAllocation<typename KernelTraits::Gemm::ElementB> block_B;
    cutlass::DeviceAllocation<typename KernelTraits::Gemm::ElementC> block_C;
    cutlass::DeviceAllocation<typename KernelTraits::Gemm::EpilogueOutputOp::ElementOutput> block_D;
    cutlass::DeviceAllocation<typename KernelTraits::Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
}

int main(int argc, char const** args)
{
    // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
    // and must have compute capability at least 90.

    if (__CUDACC__VER_MAJOR__ < 12)
    {
        std::cerr << "This example requires CUDA 12 or newer. \n"
        return 0;
    }

    cudaDeviceProp props;
    int current_device_id;
    CUDA_CHECK(cudaGetDevice(&current_device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (props.major != 9 || props.minor != 0) {
      std::cerr
        << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
      return 0;
    }

    //
    // Parse options
    //

    Options options;

    options.parse(argc, args);

    if (options.help) {
      options.print_usage(std::cout) << std::endl;
      return 0;
    }

    //
    // Evaluate CUTLASS kernels
    //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    c101::WarpSpecialGemm<c101::KernelTraits> gemm;
    gemm.run(options);
#endif

    return 0;
}