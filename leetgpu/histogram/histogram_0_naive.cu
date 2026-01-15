#include <cuda_runtime.h>

// naive写法
// 19.95ms

#define BLOCK_DIM 64

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N)
        atomicAdd(&histogram[input[idx]], 1);
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x);
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N, num_bins);

    cudaDeviceSynchronize();
}
