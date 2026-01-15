#include <cuda_runtime.h>
#include <stdio.h>

// 最naive的tree-reduction
// tree-reduction的版本上让每个线程访问相邻元素，减少分支发散和内存访问事务
// 0.33ms

#define BLOCK_DIM 128

__global__ void reduce_kernel(const float* input, float* output, int N)
{
    int idx_start = blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x;
    __shared__ float input_s[BLOCK_DIM];

    if(idx_start + local_idx < N){
        input_s[local_idx] = input[idx_start + local_idx];
    }else{
        input_s[local_idx] = 0.0f;
    }
    __syncthreads();

    for(int step = blockDim.x/2; step > 0; step /= 2){
        if(threadIdx.x < step){
            input_s[local_idx] += input_s[local_idx + step];
        }
        __syncthreads();
    }

    if(local_idx == 0)
        atomicAdd(output, input_s[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N)
{
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid((N+threadsPerBlock.x-1)/threadsPerBlock.x);
    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
