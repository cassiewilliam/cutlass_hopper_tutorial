/*
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// All2All with CUDA Streams
void all2all_with_streams(float *send_buffers[], float *recv_buffers[], int chunk_size, int num_gpus) {
    cudaStream_t streams[num_gpus][num_gpus];

    // Step 1: Initialize streams
    for (int src = 0; src < num_gpus; ++src) {
        CHECK_CUDA(cudaSetDevice(src));
        for (int dst = 0; dst < num_gpus; ++dst) {
            CHECK_CUDA(cudaStreamCreate(&streams[src][dst]));
        }
    }

    // Step 2: Perform peer-to-peer communication using streams
    for (int src = 0; src < num_gpus; ++src) {
        CHECK_CUDA(cudaSetDevice(src));
        for (int dst = 0; dst < num_gpus; ++dst) {
            if (src != dst) {
                // Asynchronously send data from src to dst
                CHECK_CUDA(cudaMemcpyPeerAsync(
                    recv_buffers[dst] + src * chunk_size, // Destination buffer on dst GPU
                    dst,                                  // Destination GPU
                    send_buffers[src] + dst * chunk_size, // Source buffer on src GPU
                    src,                                  // Source GPU
                    chunk_size * sizeof(float),           // Size of the data to send
                    streams[src][dst]                     // Use the dedicated stream
                ));
            }
        }
    }

    // Step 3: Synchronize all streams
    for (int src = 0; src < num_gpus; ++src) {
        CHECK_CUDA(cudaSetDevice(src));
        for (int dst = 0; dst < num_gpus; ++dst) {
            CHECK_CUDA(cudaStreamSynchronize(streams[src][dst]));
        }
    }

    // Step 4: Destroy streams
    for (int src = 0; src < num_gpus; ++src) {
        CHECK_CUDA(cudaSetDevice(src));
        for (int dst = 0; dst < num_gpus; ++dst) {
            CHECK_CUDA(cudaStreamDestroy(streams[src][dst]));
        }
    }
}

int main() {
    const int num_gpus = 4;
    const int chunk_size = 1024; // Size of each data chunk per GPU
    const int total_size = chunk_size * num_gpus;

    // Allocate send and receive buffers on each GPU
    float *send_buffers[num_gpus];
    float *recv_buffers[num_gpus];

    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&send_buffers[i], total_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recv_buffers[i], total_size * sizeof(float)));

        // Initialize send buffers with some values
        float *host_buffer = new float[total_size];
        for (int j = 0; j < total_size; ++j) {
            host_buffer[j] = i * 1000 + j; // Example data
        }
        CHECK_CUDA(cudaMemcpy(send_buffers[i], host_buffer, total_size * sizeof(float), cudaMemcpyHostToDevice));
        delete[] host_buffer;
    }

    // Perform All2All communication
    all2all_with_streams(send_buffers, recv_buffers, chunk_size, num_gpus);

    // Verify results
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        float *host_recv_buffer = new float[total_size];
        CHECK_CUDA(cudaMemcpy(host_recv_buffer, recv_buffers[i], total_size * sizeof(float), cudaMemcpyDeviceToHost));

        printf("GPU %d received data:\n", i);
        for (int j = 0; j < total_size; ++j) {
            printf("%.0f ", host_recv_buffer[j]);
        }
        printf("\n");
        delete[] host_recv_buffer;
    }

    // Free memory
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(send_buffers[i]));
        CHECK_CUDA(cudaFree(recv_buffers[i]));
    }

    return 0;
}
//*/
///*
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void memcpy_task(int src, int dst, int chunk_size,
                 float *send_buffer, float *recv_buffer, cudaStream_t stream) {
    // 执行 Peer-to-Peer 内存复制
    CHECK_CUDA(cudaMemcpyPeerAsync(
        recv_buffer, dst,         // 目标缓冲区和目标 GPU
        send_buffer, src,         // 源缓冲区和源 GPU
        chunk_size * sizeof(float), // 数据大小
        stream                    // 使用的 CUDA Stream
    ));

    // 等待当前 Stream 完成
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

int main() {
    const int num_gpus = 4;       // GPU 数量
    const int chunk_size = 1024 * 1024 * 128;  // 每个 chunk 的大小
    const int total_size = chunk_size * num_gpus;

    // 为每个 GPU 分配发送和接收缓冲区
    float *send_buffers[num_gpus];
    float *recv_buffers[num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&send_buffers[i], total_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recv_buffers[i], total_size * sizeof(float)));
    }

    // 创建用于每个任务的 CUDA Streams
    cudaStream_t streams[num_gpus][num_gpus];
    for (int src = 0; src < num_gpus; ++src) {
        CHECK_CUDA(cudaSetDevice(src));
        for (int dst = 0; dst < num_gpus; ++dst) {
            CHECK_CUDA(cudaStreamCreate(&streams[src][dst]));
        }
    }

    // 启动多线程进行完全并发
    std::vector<std::thread> threads;
    for (int src = 0; src < num_gpus; ++src) {
        for (int dst = 0; dst < num_gpus; ++dst) {
            if (src != dst) { // 避免自通信
                threads.emplace_back(
                    memcpy_task,
                    src, dst, chunk_size,
                    send_buffers[src] + dst * chunk_size,
                    recv_buffers[dst] + src * chunk_size,
                    streams[src][dst]
                );
            }
        }
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        t.join();
    }

    // 清理资源
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; ++j) {
            CHECK_CUDA(cudaStreamDestroy(streams[i][j]));
        }
        CHECK_CUDA(cudaFree(send_buffers[i]));
        CHECK_CUDA(cudaFree(recv_buffers[i]));
    }

    printf("All tasks completed.\n");
    return 0;
}
//*/