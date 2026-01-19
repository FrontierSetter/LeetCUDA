#include <cuda_runtime.h>
#include <stdio.h>

// 最naive的tree-reduction
// tree-reduction的版本上让每个线程访问相邻元素，减少分支发散和内存访问事务
// 增加coarse逻辑，让最开始的加载部分承担更多的计算
//      benefit：减少硬件串行化（因为block数量大于sm的数量）的开销
//      https://www.bilibili.com/video/BV1bw4m1k7a1?t=4506.7&p=10
// 增加warp-shuffle，相比于share-memory版本减少同步开销且register更快
// 0.09ms

#define BLOCK_DIM 128
#define COARSE_FACTOR 4
#define WARP_SIZE 32

__global__ void reduce_kernel(const float* input, float* output, int N)
{
    int idx_start = blockIdx.x * blockDim.x * COARSE_FACTOR;
    int local_idx = threadIdx.x;
    __shared__ float input_s[BLOCK_DIM];

    
    if(idx_start + local_idx < N){
        input_s[local_idx] = input[idx_start + local_idx];
    }else{
        input_s[local_idx] = 0.0f;
    }
    for(int i = 1; i < COARSE_FACTOR; ++i){
        int target_idx = idx_start + local_idx + i*blockDim.x;
        if(target_idx < N){
            input_s[local_idx] += input[target_idx];
        }
    }
    __syncthreads();

    // warp-shuffle
    for(int step = blockDim.x/2; step > WARP_SIZE; step /= 2){
        if(threadIdx.x < step){
            input_s[local_idx] += input_s[local_idx + step];
        }
        __syncthreads();
    }

    float sum;
    if(local_idx < WARP_SIZE){
        sum = input_s[local_idx] + input_s[local_idx+WARP_SIZE];

        for(int step = WARP_SIZE/2; step > 0; step /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, step);
        }
    }

    if(local_idx == 0)
        atomicAdd(output, sum);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid((N+threadsPerBlock.x*COARSE_FACTOR-1)/(threadsPerBlock.x*COARSE_FACTOR));
    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
