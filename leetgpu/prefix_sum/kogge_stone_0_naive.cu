#include <cuda_runtime.h>
#include <stdio.h>

// https://www.bilibili.com/video/BV1bw4m1k7a1?t=1216.7&p=11
// BLOCK_DIM = 4, 1.48ms
// BLOCK_DIM = 512, 6.10ms
// BLOCK_DIM = 128, 3.68ms

#define BLOCK_DIM 512

__global__ void scan_kernel(const float* input, float* output, int N, float* partial_sum)
{
    int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    extern __shared__ float input_s[];

    // Load data into shared memory
    if(global_idx < N){
        input_s[threadIdx.x] = input[global_idx];
    }else{
        input_s[threadIdx.x] = 0.0f;
    }
    __syncthreads();  //! 别漏了

    // Kogge-Stone parallel prefix sum
    for(int step = 1; step < blockDim.x; step *= 2){
        float tmp = 0.0f;
        if(threadIdx.x >= step){
            tmp = input_s[threadIdx.x - step];
        }
        __syncthreads();
        if(threadIdx.x >= step){
            input_s[threadIdx.x] += tmp;
        }
        __syncthreads();
    }

    // Write results and partial sums
    if(global_idx < N){
        output[global_idx] = input_s[threadIdx.x];
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
    scan_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(input, output, N, partialSum_d);

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

