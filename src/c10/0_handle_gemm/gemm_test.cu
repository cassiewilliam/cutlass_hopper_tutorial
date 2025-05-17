#include "9_cutlass_gemm.h"

int main(int argc, char** argv)
{
    int M = testM;
    int N = testN;
    int K = testK;
    if (argc > 1)
    {
        assert((argc - 1) % 2 == 0);
        for (int i = 1; i < argc; i += 2)
        {
            char*       key   = argv[i];
            char*       value = argv[i + 1];
            std::string keys(key);
            if (keys == "M")
            {
                M = std::atoi(value);
            }
            else if (keys == "N")
            {
                N = std::atoi(value);
            }
            else if (keys == "K")
            {
                K = std::atoi(value);
            }
        }
    }
    std::cout << "Testing shape M=" << M << ", N=" << N << ", K=" << K << "\n";
    using AType     = half_t;
    using BType     = half_t;
    using CType     = half_t;
    using AccumType = float;
    AccumType alpha = 1.0;
    AccumType beta  = 0.0;

    std::vector<int> AShape = {M, K};
    std::vector<int> BShape = {N, K};
    std::vector<int> CShape = {M, N};
    auto             hA     = alloc_cpu_tensor<AType>(AShape);
    random_fill(hA, AShape);
    // constant_fill(hA, AShape, (AType)1.0);
    auto hB = alloc_cpu_tensor<BType>(BShape);
    random_fill(hB, BShape);
    // constant_fill(hB, BShape, (BType)1.0);
    auto hC = alloc_cpu_tensor<CType>(CShape);
    random_fill(hC, CShape);
    // constant_fill(hC, CShape, (CType)(-13.0));
    auto goldenC = alloc_cpu_tensor<CType>(CShape);
    random_fill(goldenC, CShape);
    auto dA  = alloc_gpu_tensor<AType>(AShape);
    auto dB  = alloc_gpu_tensor<BType>(BShape);
    auto dgC = alloc_gpu_tensor<CType>(CShape);
    auto dC  = alloc_gpu_tensor<CType>(CShape);

    /// timers
    CPUTimer cpu_timer;
    GPUTimer gpu_timer;

    /// copy data
    std::cout << "Copying data from CPU to GPU...\n";
    cpu_timer.tick();
    copy_to_gpu(hA, dA, AShape);
    copy_to_gpu(hB, dB, BShape);
    copy_to_gpu(hC, dC, CShape);
    copy_to_gpu(goldenC, dgC, CShape);
    cpu_timer.tick();
    std::cout << "Copy data done! Use " << cpu_timer.report_last_ms() << " ms.\n";

    /// compute gpu reference
    std::cout << "Computing gpu reference values...\n";
    GemmParams gpu_params(M, N, K, dA, dB, dgC, alpha, beta);
    gpu_timer.sync_all();
    gpu_timer.tick();
    reference_gpu_gemm(gpu_params);
    gpu_timer.tick();
    gpu_timer.sync_all();
    std::cout << "GPU reference done! Use " << gpu_timer.report_last_ms() << " ms.\n";

    /// copy results
    std::cout << "Copying results...\n";
    copy_to_cpu(goldenC, dgC, CShape);
    std::cout << "Copying results done!\n";

    /// compute gpu kernel
    std::cout << "Computing gpu kernel values...\n";
    GemmParams gpu_kernel_params(M, N, K, dA, dB, dC, alpha, beta);
    gpu_gemm(gpu_kernel_params, true);
    std::cout << "GPU kernel done!\n";

    /// copy results

    std::cout << "Copying results...\n";
    copy_to_cpu(hC, dC, CShape);
    std::cout << "Copying results done!\n";

    /// compare results
    assert_allclose(hC, goldenC, CShape, /*rtol=*/1e-3, /*dump=*/false);
    std::cout << "Correct!\n";

    /// profile
    std::cout << "Profile performance...\n";
    gpu_timer.sync_all();
    gpu_timer.tick();
    for (int i = 0; i < iters; ++i)
    {
        gpu_gemm(gpu_params, /*verbose=*/false);
    }
    gpu_timer.tick();
    gpu_timer.sync_all();
    float latency = gpu_timer.report_last_ms() / float(iters);
    std::cout << "Profile done! Average latency (ms) is " << latency << "\n";
    std::cout << "TFLOPS: " << ((double)M * (double)N * (double)K * 2.0) / (latency / 1000.0) / 1e12
              << "\n";

    free_cpu_tensor(hA);
    free_cpu_tensor(hB);
    free_cpu_tensor(hC);
    free_cpu_tensor(goldenC);
    free_gpu_tensor(dA);
    free_gpu_tensor(dB);
    free_gpu_tensor(dC);
    return 0;
}