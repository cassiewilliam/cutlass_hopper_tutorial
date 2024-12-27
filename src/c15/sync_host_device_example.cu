#include <stdio.h>
#include <cuda_runtime.h>

// Release 指令，带 release 修饰符
__device__ void ld_release(int *ptr, int value) {
    asm volatile("st.global.release.sys.b32 [%0], %1;\n" :: "l"(ptr), "r"(value));
}

// Acquire 指令，带 acquire 修饰符
__device__ int ld_acquire(int *ptr) {
    int state = 0;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
    return state;
}

// 设备函数：使用 acquire-release 模式
__global__ void acquire_release_example(int *data, int *flag) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // 等待 flag 从主机设置为 1
        while (ld_acquire(flag) == 0); // 等待主机设置 flag
        int value = data[0];          // 消费者读取数据
        printf("Thread %d read data: %d\n", tid, value);
    }
    // else if (tid == 1) {
    //     // 生产者：写入数据并设置 flag（本例中未触发）
    //     data[0] = 42;  // 写入数据
    //     __threadfence_system();  // 确保写入对全局可见
    //     ld_release(flag, 1);     // 使用 release 语义设置 flag
    // }
}

int main() {
    int *d_data, *d_flag;
    int h_flag = 0;
    int h_data = 123; // 数据初始值

    // 分配设备内存
    cudaMalloc(&d_data, sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));

    // 初始化设备数据
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_flag, 0, sizeof(int)); // 将设备 flag 初始化为 0

    // 创建两个用户定义的 CUDA 流
    cudaStream_t kernel_stream, memcpy_stream;
    cudaStreamCreate(&kernel_stream);
    cudaStreamCreate(&memcpy_stream);

    // 启动内核函数到 kernel_stream
    acquire_release_example<<<1, 1, 0, kernel_stream>>>(d_data, d_flag);

    // 主机侧模拟延迟，异步设置 flag
    printf("Host is setting the flag...\n");
    h_flag = 1; // 主机设置 flag 值
    cudaMemcpyAsync(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice, memcpy_stream); // 异步拷贝到设备

    // 等待所有操作完成
    cudaStreamSynchronize(kernel_stream);
    cudaStreamSynchronize(memcpy_stream);

    // 释放资源
    cudaStreamDestroy(kernel_stream);
    cudaStreamDestroy(memcpy_stream);
    cudaFree(d_data);
    cudaFree(d_flag);

    return 0;
}