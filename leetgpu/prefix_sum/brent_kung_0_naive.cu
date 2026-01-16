#include <cuda_runtime.h>
#include <stdio.h>

// https://www.bilibili.com/video/BV1bw4m1k7a1?t=1553.5&p=12
// BLOCK_DIM = 512, 11.38ms

#define BLOCK_DIM 512

__global__ void scan_kernel(const float* input, float* output, int N, float* partial_sum)
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int local_idx = threadIdx.x;
    __shared__ float input_s[BLOCK_DIM];

    if(global_idx < N){
        input_s[local_idx] = input[global_idx];
    }else{
        input_s[local_idx] = 0.0f;
    }
    __syncthreads();

    // reduction
    int begin_idx = 0, step = 1, stride = 2;
    while(begin_idx+step < BLOCK_DIM){
        int left_idx = begin_idx + stride * local_idx;
        int right_idx = left_idx + step;

        if(right_idx < BLOCK_DIM){
            input_s[right_idx] += input_s[left_idx];
        }
        // __syncthreads();     //! 放在这个位置就会有精度误差，why？

        begin_idx = begin_idx + step;
        step *= 2;
        stride *= 2;
        __syncthreads();
    }

    // post reduction
    stride /= 2;
    step /= 2;
    begin_idx = begin_idx - step;
    stride /= 2;
    step /= 2;
    while(begin_idx >= 1){
        int left_idx = begin_idx + stride * local_idx;
        int right_idx = left_idx + step;

        if(right_idx < BLOCK_DIM){
            input_s[right_idx] += input_s[left_idx];
        }
        // __syncthreads();    //! 放在这个位置就会有精度误差，why？

        begin_idx = begin_idx - step;
        stride /= 2;
        step /= 2;
        __syncthreads();
    }

    // Write results and partial sums
    if(global_idx < N){
        output[global_idx] = input_s[local_idx];
    }

    if(partial_sum && threadIdx.x == blockDim.x-1){
        partial_sum[blockIdx.x] = input_s[threadIdx.x];
    }

}

__global__ void add_kernel(const float* partial_sum, float* output, int N)
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(global_idx < N && blockIdx.x > 0)
        output[global_idx] += partial_sum[blockIdx.x - 1];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N)
{
    const unsigned int threadsPerBlock = BLOCK_DIM;
    const unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *partialSum_d;
    cudaMalloc((void**) &partialSum_d, blocksPerGrid * sizeof(float));

    // scan kernel - first pass
    scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, partialSum_d);

    if(blocksPerGrid > 1){
        // Allocate temporary buffer for recursive scan
        float *temp_partial;
        cudaMalloc((void**) &temp_partial, blocksPerGrid * sizeof(float));

        // Recursive scan on partial sums
        solve(partialSum_d, temp_partial, blocksPerGrid);

        // Add kernel - add prefix sums to each block
        add_kernel<<<blocksPerGrid, threadsPerBlock>>>(temp_partial, output, N);

        // Clean up temporary buffer
        cudaFree(temp_partial);
    }

    cudaDeviceSynchronize();
    cudaFree(partialSum_d);
}

