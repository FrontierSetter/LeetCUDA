#include <cuda_runtime.h>

#define TILE_SIZE 16
#define COARSE_FACTOR 4

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    int col_start = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    __shared__ float A_s[TILE_SIZE][TILE_SIZE];
    __shared__ float B_s[TILE_SIZE][TILE_SIZE];

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // float sum = 0.0f;
    float sum_c[COARSE_FACTOR];
    for(int c = 0; c < COARSE_FACTOR; ++c){
        sum_c[c] = 0.0f;
    }
    for(int t_i = 0; t_i < numTiles; ++t_i){
        int A_r = row;
        int A_c = t_i * TILE_SIZE + threadIdx.x;
        if(A_r < M && A_c < N){
            A_s[threadIdx.y][threadIdx.x] = A[A_r * N + A_c];
        }else{
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        for(int c = 0; c < COARSE_FACTOR; ++c){
            int B_r = t_i * TILE_SIZE + threadIdx.y;
            int B_c = col_start + c * TILE_SIZE;
            if(B_r < N && B_c < K){
                B_s[threadIdx.y][threadIdx.x] = B[B_r * K + B_c];
            }else{
                B_s[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            for(int i = 0; i < TILE_SIZE; ++i){
                sum_c[c] += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for(int c = 0; c < COARSE_FACTOR; ++c){
        int col = col_start + c * TILE_SIZE;
        if(row < M && col < K){
            C[row*K + col] = sum_c[c];
        }
    }
    
    return;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(((K + threadsPerBlock.x - 1) / threadsPerBlock.x + COARSE_FACTOR - 1) / COARSE_FACTOR,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
