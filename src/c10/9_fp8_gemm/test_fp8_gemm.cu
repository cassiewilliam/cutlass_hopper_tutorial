
#include "include/fp8_gemm_utils.h"
#include "src/c10/9_fp8_gemm/0_fp8_warp_specialization_gemm/fp8_warp_specialization_gemm.cuh"
#include "src/c10/9_fp8_gemm/1_deep_gemm_w8a8/deep_gemm_runner.h"

/// Initialization

namespace c109
{

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;

StrideA stride_A;
StrideB stride_B;
StrideD stride_D;
uint64_t seed;

cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D_ref;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D_deepgemm;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options<RasterOrderOptions> &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto d_coord = cutlass::make_Coord(options.m * options.l, options.n);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_D.resize(d_coord);
  tensor_D_ref.resize(d_coord);
  tensor_D_deepgemm.resize(d_coord);

  initialize_tensor(tensor_A.host_view(), seed + 2022);
  initialize_tensor(tensor_B.host_view(), seed + 2023);

  std::fill(tensor_D.host_data(), tensor_D.host_data() + tensor_D.capacity(), 0);
  std::fill(tensor_D_ref.host_data(), tensor_D_ref.host_data() + tensor_D_ref.capacity(), 0);
  std::fill(tensor_D_deepgemm.host_data(), tensor_D_deepgemm.host_data() + tensor_D_deepgemm.capacity(), 0);

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_D.sync_device();
  tensor_D_deepgemm.sync_device();

  if (options.device_scale) {
    scalar_alpha.resize(cutlass::make_Coord(1));
    scalar_beta.resize(cutlass::make_Coord(1));
    scale_A.resize(cutlass::make_Coord(1));
    scale_B.resize(cutlass::make_Coord(1));
    scale_D.resize(cutlass::make_Coord(1));

    cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    cutlass::reference::host::TensorFill(scale_A.host_view(), options.scale_a);
    cutlass::reference::host::TensorFill(scale_B.host_view(), options.scale_b);
    cutlass::reference::host::TensorFill(scale_D.host_view(), options.scale_d);

    scalar_alpha.sync_device();
    scalar_beta.sync_device();
    scale_A.sync_device();
    scale_B.sync_device();
    scale_D.sync_device();
  }
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options<RasterOrderOptions> &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B},
    {
      {}, // epilogue.thread
      nullptr, stride_D,
      tensor_D.device_data(), stride_D
    }
  };

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = options.alpha;
  fusion_args.beta = options.beta;
  fusion_args.alpha_ptr = scalar_alpha.device_data();
  fusion_args.beta_ptr = scalar_beta.device_data();
  fusion_args.scale_a = options.scale_a;
  fusion_args.scale_b = options.scale_b;
  fusion_args.scale_a_ptr = scale_A.device_data();
  fusion_args.scale_b_ptr = scale_B.device_data();

  // leaving/setting these as nullptr disables the fusion at runtime
  fusion_args.bias_ptr = nullptr;

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options<RasterOrderOptions> &options, std::string name = "cutlass") {
  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto B = cute::make_tensor(tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto D = cute::make_tensor(tensor_D_ref.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  using unused_t = decltype(D);

  cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

  cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      unused_t,
      decltype(D)> epilogue_params;

  epilogue_params.D = D;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;
  epilogue_params.scale_a = options.scale_a;
  epilogue_params.scale_b = options.scale_b;
  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  bool passed = true;
  if (name == "cutlass")
  {
    tensor_D.sync_host();
    passed = cutlass::reference::host::TensorEquals(tensor_D_ref.host_view(), tensor_D.host_view());

    if (!passed)
    {
      compare_tensors(tensor_D_ref, tensor_D);
    }
  }
  else if (name == "deepgemm")
  {
    tensor_D_deepgemm.sync_host();
    passed = cutlass::reference::host::TensorEquals(tensor_D_ref.host_view(), tensor_D_deepgemm.host_view());

    if (!passed)
    {
      compare_tensors(tensor_D_ref, tensor_D_deepgemm);
    }
  }

  return passed;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/// Execute a given example GEMM computation
template <typename Gemm>
int run_gemm(Options<RasterOrderOptions> &options)
{
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

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}


/// Execute a given example GEMM computation
int run_deepgemm_fp8_gemm(Options<RasterOrderOptions> &options)
{
    auto gemm_runner = std::make_shared<c108::DeepGemmRunner>();

    deep_gemm::GemmType gemm_type = deep_gemm::GemmType::PerTensorQuant;
    gemm_runner->tunning(options.m, options.n, options.k, 1, gemm_type);

    gemm_runner->per_tensor_gmm((half *)(tensor_D_deepgemm.device_data()), // half*                res,
                                options.m,                                 // int                  m,
                                options.n,                                 // int                  n,
                                options.k,                                 // int                  k,
                                options.alpha,                             // const float&         alpha,
                                options.beta,                              // const float&         beta,
                                (__nv_fp8_e4m3 *)(tensor_A.device_data()), // const __nv_fp8_e4m3* input,
                                (__nv_fp8_e4m3 *)(tensor_B.device_data()), // const __nv_fp8_e4m3* kernel,
                                &options.scale_a,                          // const float&         input_scale,
                                &options.scale_b,                          // const float&         kernel_scale,
                                0);                                        // cudaStream_t         stream

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    Result result;
    result.passed = verify(options, "deepgemm");

    std::cout << "  deepgemm w8a8 fp8 gemm Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    if (!result.passed) {
        exit(-1);
    }

    // Run profiling loop
    if (options.iterations > 0)
    {
        GpuTimer timer;
        timer.start();
        for (int iter = 0; iter < options.iterations; ++iter)
        {
            gemm_runner->per_tensor_gmm((half *)(tensor_D_deepgemm.device_data()), // half*                res,
                                        options.m,                                 // int                  m,
                                        options.n,                                 // int                  n,
                                        options.k,                                 // int                  k,
                                        options.alpha,                             // const float&         alpha,
                                        options.beta,                              // const float&         beta,
                                        (__nv_fp8_e4m3 *)(tensor_A.device_data()), // const __nv_fp8_e4m3* input,
                                        (__nv_fp8_e4m3 *)(tensor_B.device_data()), // const __nv_fp8_e4m3* kernel,
                                        &options.scale_a,                           // const float&         input_scale,
                                        &options.scale_b,                           // const float&         kernel_scale,
                                        0);                                        // cudaStream_t         stream
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

        std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
        std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
        std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
        std::cout << "  GFLOPS: " << result.gflops << std::endl;
    }

    return 0;
}

} // namespace 109

int main(int argc, char const** args)
{
    // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
    // and must have compute capability at least 90.

    if (__CUDACC_VER_MAJOR__ < 12)
    {
        std::cerr << "This example requires CUDA 12 or newer. \n";
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

    Options<RasterOrderOptions> options;

    options.parse(argc, args);

    if (options.help) {
      options.print_usage(std::cout) << std::endl;
      return 0;
    }

    //
    // Evaluate CUTLASS kernels
    //

    c109::initialize(options);

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    c109::run_gemm<c109::Gemm>(options);
    c109::run_deepgemm_fp8_gemm(options);
#endif

    return 0;
}