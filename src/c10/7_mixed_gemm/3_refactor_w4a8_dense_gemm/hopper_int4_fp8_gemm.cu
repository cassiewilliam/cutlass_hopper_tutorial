#include "cli_options.h"
#include "hopper_int4_fp8_gemm_kernel_launch.h"
#include "kernel_traits.h"
#include "tile_scheduler.hpp"

#include "cutlass/cutlass.h"

#include <iostream>

int main(int argc, char const* argv[])
{
    cudaDeviceProp props;
    cudaError_t    error = cudaGetDeviceProperties(&props, 0);

    if (error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error)
                  << std::endl;
        return -1;
    }

    if (props.major != 9)
    {
        std::cout << "This example requires NVIDIA's Hopper Architecture GPU with compute "
                     "capability 90a\n"
                  << std::endl;
        return 0;
    }

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    std::cout
        << "This example requires NVIDIA's Hopper Architecture GPU with compute capability 90a\n"
        << std::endl;
    return 0;
#endif

    Options options;

    options.parse(argc, argv);

    if (options.help)
    {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    if (options.error)
    {
        std::cerr << "Aborting execution." << std::endl;
        return -1;
    }

    if (options.seed == 0)
        srand(time(NULL));
    else
        srand(options.seed);

    
}