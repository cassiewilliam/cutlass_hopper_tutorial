#include <cstdio>
#include <cuda_runtime.h>
#include <cute/util/print.hpp>

#define NumThreadsPerWarp 32

__device__ int canonical_warp_idx_sync() { 
    return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarp, 0);
}

__device__ int canonical_warp_idx() { 
    return threadIdx.x / NumThreadsPerWarp;
}

__global__ void divergence_test() {
    int warp_id_sync = canonical_warp_idx_sync();
    int warp_id_no_sync = canonical_warp_idx();

    // 故意制造发散
    if (threadIdx.x % 2 == 0) {
        printf("Even Thread %d -> warp_id_sync: %d, warp_id_no_sync: %d\n", threadIdx.x, warp_id_sync, warp_id_no_sync);
    } else {
        printf("Odd Thread %d -> warp_id_sync: %d, warp_id_no_sync: %d\n", threadIdx.x, warp_id_sync, warp_id_no_sync);
    }
}

int main() {
    printf("run c0_test_warp_index \n");
    // 启动内核
    divergence_test<<<1, 64>>>(); // 启动 64 个线程（2 个 warp）

    // 等待 GPU 任务完成
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // 确保 CUDA 设备资源完全释放
    cudaDeviceReset();
}