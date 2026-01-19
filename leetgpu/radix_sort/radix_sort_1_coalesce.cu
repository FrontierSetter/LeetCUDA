#include <cuda_runtime.h>
#include <stdio.h>

// naive实现基数排序
// 耗时388.79ms

# define BLOCK_DIM 512

__global__ void printArr_kernel(const unsigned int* input, int N)
{
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("[block-%d]: ", blockIdx.x);
        for(int i = 0; i < N; ++i){
            printf("%u, ", input[i]);
        }
        printf("\n");
    }
}

__global__ void scan_kernel(const unsigned int* input, unsigned int* output, int N, unsigned int* partialSum)
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int local_idx = threadIdx.x;

    __shared__ unsigned int input_s[BLOCK_DIM];

    //? exclusive的scan会导致在rearrange的时候难以直接得到「全部的1的数量」
    // if(global_idx-1 >= 0 && global_idx-1 < N){
    //     input_s[local_idx] = input[global_idx-1];
    // }else{
    //     input_s[local_idx] = 0.0f;
    // }

    if(global_idx < N){
        input_s[local_idx] = input[global_idx];
    }else{
        input_s[local_idx] = 0;
    }
    __syncthreads();

    int begin_idx = 0, step = 1, stride = 2;
    while(begin_idx+step < blockDim.x){
        int left_idx = begin_idx + local_idx * stride;
        int right_idx = left_idx + step;
        if(right_idx < blockDim.x){
            input_s[right_idx] += input_s[left_idx]; 
        }

        begin_idx += step;
        step *= 2;
        stride *= 2;
        __syncthreads();
    }

    step /= 2;
    stride /= 2;
    begin_idx -= step;
    step /= 2;
    stride /= 2;

    while(begin_idx >= 1){
        int left_idx = begin_idx + local_idx * stride;
        int right_idx = left_idx + step;
        if(right_idx < blockDim.x){
            input_s[right_idx] += input_s[left_idx]; 
        }

        begin_idx -= step;
        step /= 2;
        stride /= 2;
        __syncthreads();
    }

    if(global_idx < N){
        output[global_idx] = input_s[local_idx];
    }
    if(local_idx == blockDim.x-1){
        partialSum[blockIdx.x] = input_s[local_idx];
    }
}

__global__ void add_kernel(unsigned int* output, int N, unsigned int* partialSum)
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(blockIdx.x > 0 && global_idx < N){
        output[global_idx] += partialSum[blockIdx.x-1];
    }
}

void scan(const unsigned int* input, unsigned int* output, int N)
{
    unsigned int threadsPerBlock = BLOCK_DIM;
    unsigned int blocksPerGrid = (N+threadsPerBlock-1)/ threadsPerBlock;
    unsigned int *partialSum_d;
    cudaMalloc((void**)&partialSum_d, sizeof(unsigned int)*blocksPerGrid);

    scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, partialSum_d);

    // printf("scan-output:\n");
    // printArr_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N);
    // cudaDeviceSynchronize();
    // printf("scan-partialSum_d:\n");
    // printArr_kernel<<<blocksPerGrid, threadsPerBlock>>>(partialSum_d, blocksPerGrid);
    // cudaDeviceSynchronize();

    if(blocksPerGrid > 1){
        scan(partialSum_d, partialSum_d, blocksPerGrid);
        add_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N, partialSum_d);
    }

    cudaFree(partialSum_d);
    cudaDeviceSynchronize();
}

// 先做内部的基数排序
__global__ void getOne_kernel(unsigned int* input, unsigned int* output_one, int N, const unsigned int bitMask)
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int local_idx = threadIdx.x;
    __shared__ unsigned int input_s[BLOCK_DIM];

    if(global_idx < N){
        input_s[local_idx] = input[global_idx];
    }else{
        input_s[local_idx] = bitMask;
    }
    __syncthreads();
    // TODO: 卡在这里了

    if(global_idx < N){
        if(input[global_idx] & bitMask){
            output_one[global_idx] = 1;
        }else{
            output_one[global_idx] = 0;
        }
    }
}

__global__ void rearrange_kernel(const unsigned int* input, const unsigned int* oneArr, \
    const unsigned int* oneArr_prefixSum, unsigned int* output, int N)
{
    int total_ones = oneArr_prefixSum[N-1];
    int total_zeros = N - total_ones;

    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(global_idx >= N)
        return;

    int prefix_ones = 0;
    if(global_idx > 0)
        prefix_ones = oneArr_prefixSum[global_idx-1];
    int new_idx;
    if(oneArr[global_idx]){ // oneArr[global_idx] == 1
        new_idx = total_zeros + prefix_ones;
    }else{  // oneArr[global_idx] == 0
        new_idx = global_idx - prefix_ones;
    }

    output[new_idx] = input[global_idx];
}

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) 
{
    unsigned int threadsPerBlock = BLOCK_DIM;
    unsigned int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock;

    unsigned int* input_shadow;
    cudaMalloc((void**)&input_shadow, N*sizeof(unsigned int));
    cudaMemcpy(input_shadow, input, N*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    unsigned int* output_shadow;
    cudaMalloc((void**)&output_shadow, N*sizeof(unsigned int));

    unsigned int bitMask = 1;
    unsigned int *oneArr;
    cudaMalloc((void**)&oneArr, N*sizeof(unsigned int));
    unsigned int *oneArr_prefixSum;
    cudaMalloc((void**)&oneArr_prefixSum, N*sizeof(unsigned int));
    for(int i = 0; i < 32; ++i){
        // printf("input:\n");
        // printArr_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_shadow, N);
        // cudaDeviceSynchronize();
        // 产生01数组
        getOne_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_shadow, oneArr, N, bitMask);
        cudaDeviceSynchronize();
        // printf("oneArr:\n");
        // printArr_kernel<<<blocksPerGrid, threadsPerBlock>>>(oneArr, N);
        // cudaDeviceSynchronize();

        // scan：01数组到prefix-sum
        scan(oneArr, oneArr_prefixSum, N);
        // printf("oneArr_prefixSum:\n");
        // printArr_kernel<<<blocksPerGrid, threadsPerBlock>>>(oneArr_prefixSum, N);
        // cudaDeviceSynchronize();

        // 重排
        rearrange_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_shadow, oneArr, oneArr_prefixSum, output_shadow, N);
        cudaDeviceSynchronize();

        // printf("output_shadow:\n");
        // printArr_kernel<<<blocksPerGrid, threadsPerBlock>>>(output_shadow, N);
        // cudaDeviceSynchronize();

        unsigned int* tmp = input_shadow;
        input_shadow = output_shadow;
        output_shadow = tmp;

        bitMask <<= 1;

    }
    cudaMemcpy(output, input_shadow, N*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaFree(input_shadow);
    cudaFree(output_shadow);
}

