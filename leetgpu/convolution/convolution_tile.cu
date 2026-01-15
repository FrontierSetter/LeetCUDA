#include <cuda_runtime.h>

// 以output作为tile，分多次读取需要的数据 
// TODO：以input作为tile基础，使部分线程在计算时空闲

#define TILE_SIZE 32

__constant__ float kernel_arr[1024];

__global__ void convolution_kernel(const float* input, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols)
{
    int row_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row_begin = blockIdx.y * blockDim.y;
    int col_begin = blockIdx.x * blockDim.x;

    int data_rows = TILE_SIZE + kernel_rows - 1;
    int data_cols = TILE_SIZE + kernel_cols - 1;

    __shared__ float input_s[TILE_SIZE+32][TILE_SIZE+32];

    for(int share_row = local_row; share_row < data_rows; share_row += TILE_SIZE){
        for(int share_col = local_col; share_col < data_cols; share_col += TILE_SIZE){
            int input_row = row_begin + share_row;
            int input_col = col_begin + share   _col;
            if(input_row < input_rows && input_col < input_cols){
                input_s[share_row][share_col] = input[input_row * input_cols + input_col];
            }else{
                input_s[share_row][share_col] = 0.0f;
            }
        }
    }
    __syncthreads();


    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    if(row_out >= output_rows || col_out >= output_cols)
        return;
    
    float result = 0;
    for(int r_offset = 0; r_offset < kernel_rows; ++r_offset){
        for(int c_offset = 0; c_offset < kernel_cols; ++c_offset){
            result += input_s[local_row+r_offset][local_col+c_offset] * kernel_arr[r_offset*kernel_cols + c_offset];
        }
    }

    output[row_out*output_cols + col_out] = result;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols)
{
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // cudaMemcpy(&kernel_arr[0], kernel, sizeof(float)*kernel_rows*kernel_cols, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel_arr, kernel, sizeof(float)*kernel_rows*kernel_cols);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blockPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (output_rows + threadsPerBlock.y - 1)/ threadsPerBlock.y);

    convolution_kernel<<<blockPerGrid, threadsPerBlock>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
