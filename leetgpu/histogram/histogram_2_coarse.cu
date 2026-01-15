#include <cuda_runtime.h>

// naive写法
// 线程使用私有histogram，再naive汇总（可用reduce优化）
// 使用coarse，减少开头shared memory初始化和结尾atomic提交的开销
// 3.71ms (blockdim=512)

#define BLOCK_DIM 512
#define MAX_BIN 1024
#define COARSE_FACTOR 4

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins)
{
    int idx_start = blockDim.x * blockIdx.x * COARSE_FACTOR;
    __shared__ int histogram_s[MAX_BIN];
    for(int start_idx = 0; start_idx < num_bins; start_idx += blockDim.x){
        if(start_idx+threadIdx.x < num_bins)
            histogram_s[start_idx+threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for(int i = 0; i < COARSE_FACTOR; ++i){
        int idx = idx_start + i*blockDim.x + threadIdx.x;
        if(idx < N)
            atomicAdd(&histogram_s[input[idx]], 1);
    }
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
    dim3 blocksPerGrid((N+(threadsPerBlock.x*4)-1)/(threadsPerBlock.x*4));
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N, num_bins);

    cudaDeviceSynchronize();
}
