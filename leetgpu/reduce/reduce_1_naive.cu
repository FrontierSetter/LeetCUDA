#include <cuda_runtime.h>
#include <stdio.h>

// 最naive的tree-reduction
// 0.35ms

__global__ void reduce_kernel(const float* input, float* output, int N)
{
    int idx_start = blockIdx.x * blockDim.x * 2;
    int local_idx = threadIdx.x * 2;
    __shared__ float input_s[64 * 2];

    if(idx_start + local_idx < N){
        input_s[local_idx] = input[idx_start + local_idx];
    }else{
        input_s[local_idx] = 0.0f;
    }
    if(idx_start + local_idx+1 < N){
        input_s[local_idx+1] = input[idx_start + local_idx+1];
    }else{
        input_s[local_idx+1] = 0.0f;
    }
    __syncthreads();

    for(int step = 1; step <= blockDim.x; step *= 2){
        if(threadIdx.x % step == 0){
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
    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid((((N+1)/2)+threadsPerBlock.x-1)/threadsPerBlock.x);
    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
