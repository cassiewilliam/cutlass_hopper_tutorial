#include <stdio.h>
#include <stdint.h>

static __device__ __inline__ uint32_t __smid()
{
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void block_to_sm_mapping()
{
    // 获取线程块和线程的标识符
    int block_id = blockIdx.x + blockIdx.y * gridDim.x; // 线程块的全局 ID
    int sm_id = __smid();                              // 当前线程所属的 SM ID

    // 只在 Block 的第一个线程中打印结果
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        printf("Block ID: %d, SM ID: %d\n", block_id, sm_id);
    }
}

int main() {
    // 定义网格和线程块的大小
    dim3 grid(4, 4); // 4x4 的线程块网格
    dim3 block(8, 8); // 每个线程块内有 8x8 个线程

    // 启动 CUDA 内核
    block_to_sm_mapping<<<grid, block>>>();

    // 同步并检查错误
    cudaDeviceSynchronize();
    return 0;
}