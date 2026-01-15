
#include <cuda_runtime.h>
__constant__ float c_kernel[1024];

__global__ void convolution2d(const float* input, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols){
    __shared__ float s_data[2034];
    int local_row = threadIdx.x, local_col = threadIdx.y;

    int global_row = threadIdx.x + blockDim.x * blockIdx.x;
    int global_col = threadIdx.y + blockDim.y * blockIdx.y;

    int tile_start_x = blockDim.x * blockIdx.x, tile_start_y = blockDim.y * blockIdx.y;
    int tile_size_x = blockDim.x + kernel_rows -1, tile_size_y = blockDim.y + kernel_cols -1;

    int output_size_x = input_rows - kernel_rows + 1, output_size_y = input_cols - kernel_cols + 1;
    for(int i = local_row; i < tile_size_x; i+=blockDim.x){
        for(int j = local_col; j < tile_size_y; j+=blockDim.y){
            if(tile_start_x + i < input_rows && tile_start_y + j < input_cols){
                s_data[i * tile_size_y + j] = input[(tile_start_x + i) * input_cols + (tile_start_y + j)];
            }
            else{
                s_data[i * tile_size_y + j] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    if(global_row < output_size_x && global_col < output_size_y){
        float sum=0.0f;
        for(int i=0; i<kernel_rows; i++){
            for(int j=0; j<kernel_cols; j++){
                sum += c_kernel[i*kernel_cols + j] * s_data[(local_row+i) * tile_size_y + (local_col+j)];
            }
        }
        output[global_row * output_size_y + global_col] = sum;
    }
}
// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    cudaMemcpyToSymbol(c_kernel, kernel, kernel_rows*kernel_cols*sizeof(float));
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((input_rows+16-1)/16, (input_cols+16-1)/16);
    convolution2d<<<blocksPerGrid, threadsPerBlock>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}