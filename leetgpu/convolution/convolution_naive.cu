#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols)
{
    int row_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    if(row_out >= output_rows || col_out >= output_cols)
        return;
    
    float result = 0;
    for(int r_offset = 0; r_offset < kernel_rows; ++r_offset){
        for(int c_offset = 0; c_offset < kernel_cols; ++c_offset){
            result += input[(row_out+r_offset)*input_cols + (col_out+c_offset)] * kernel[r_offset*kernel_cols + c_offset];
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
    dim3 threadsPerBlock(32, 32);
    dim3 blockPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (output_rows + threadsPerBlock.y - 1)/ threadsPerBlock.y);

    convolution_kernel<<<blockPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
