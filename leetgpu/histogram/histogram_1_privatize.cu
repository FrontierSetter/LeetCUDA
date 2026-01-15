#include <cuda_runtime.h>

// naive写法
// 线程使用私有histogram，再naive汇总（可用reduce优化）
// 23.01(blockdim=128), 84.04(blockdim=64)，13.94(blockdim=512)

#define BLOCK_DIM 512
#define MAX_BIN 1024

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int histogram_s[MAX_BIN];
    for(int start_idx = 0; start_idx < num_bins; start_idx += blockDim.x){
        if(start_idx+threadIdx.x < num_bins)
            histogram_s[start_idx+threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if(idx < N)
        atomicAdd(&histogram_s[input[idx]], 1);
    __syncthreads();

    if(threadIdx.x == 0){
        for(int i = 0; i < num_bins; ++i){
            atomicAdd(&histogram[i], histogram_s[i]);
        }
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x);
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N, num_bins);

    cudaDeviceSynchronize();
}
