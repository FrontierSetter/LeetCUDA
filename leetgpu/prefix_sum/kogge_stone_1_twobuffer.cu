#include <cuda_runtime.h>
#include <stdio.h>

// https://www.bilibili.com/video/BV1bw4m1k7a1?t=1216.7&p=11
// BLOCK_DIM = 512, 0.84ms

#define BLOCK_DIM 512

__global__ void scan_kernel(const float* input, float* output, int N, float* partial_sum)
{
    int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float input_s_1[BLOCK_DIM];
    __shared__ float input_s_2[BLOCK_DIM];

    float *buffer_in = input_s_1;
    float *buffer_out = input_s_2;

    // Load data into shared memory
    if(global_idx < N){
        buffer_in[threadIdx.x] = input[global_idx];
    }else{
        buffer_in[threadIdx.x] = 0.0f;
    }
    __syncthreads();  //! 别漏了

    // Kogge-Stone parallel prefix sum
    for(int step = 1; step < blockDim.x; step *= 2){
        if(threadIdx.x >= step){
            buffer_out[threadIdx.x] = buffer_in[threadIdx.x] + buffer_in[threadIdx.x - step];
        }else{
            buffer_out[threadIdx.x] = buffer_in[threadIdx.x];
        }
        __syncthreads();
        float *tmp = buffer_in;
        buffer_in = buffer_out;
        buffer_out = tmp;
        // __syncthreads();    //! 放在这个位置就会有精度误差，why？
    }

    // Write results and partial sums
    if(global_idx < N){
        output[global_idx] = buffer_in[threadIdx.x];
    }

    if(partial_sum && threadIdx.x == blockDim.x-1){
        partial_sum[blockIdx.x] = buffer_in[threadIdx.x];
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

