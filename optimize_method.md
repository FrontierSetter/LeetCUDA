# LeetCUDA 项目 Kernel 优化方法总结

## 📖 目录
- [1. 按难度级别的优化方法](#1-按难度级别的优化方法)
  - [1.1 ⭐️ Easy 级别 (基础操作)](#11-⭐️-easy-级别-基础操作)
  - [1.2 ⭐️⭐️ Medium 级别 (进阶优化)](#12-⭐️⭐️-medium-级别-进阶优化)
  - [1.3 ⭐️⭐️⭐️ Hard 级别 (Tensor Core)](#13-⭐️⭐️⭐️-hard-级别-tensor-core)
  - [1.4 ⭐️⭐️⭐️⭐️ Hard+ 级别 (高级优化)](#14-⭐️⭐️⭐️⭐️-hard-级别-高级优化)
  - [1.5 ⭐️⭐️⭐️⭐️⭐️ Hard++ 级别 (前沿技术)](#15-⭐️⭐️⭐️⭐️⭐️-hard-级别-前沿技术)
- [2. 核心优化技术详解](#2-核心优化技术详解)
  - [2.1 内存访问优化](#21-内存访问优化)
  - [2.2 计算优化](#22-计算优化)
  - [2.3 流水线优化](#23-流水线优化)
- [3. 性能优化通用模式](#3-性能优化通用模式)
- [4. 关键实现思路总结](#4-关键实现思路总结)
- [5. 实际性能数据](#5-实际性能数据)

## 1. 按难度级别的优化方法

### 1.1 ⭐️ Easy 级别 (基础操作)

**特征**: 直接实现算法逻辑，最小化优化，注重功能正确性

**包含的 kernels**:
- Elementwise 操作: `elementwise_add`, `elementwise_mul`
- 激活函数: `relu`, `sigmoid`, `gelu`, `swish`
- 简单数学运算: `dot_product`, `embedding`
- 基础变换: `mat_transpose`

**核心优化方法**:

#### 1.1.1 向量化访问 (Vec4)
```cpp
// 128位向量化加载存储
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);  // 128位向量化加载
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;  // 128位向量化存储
  }
}
```

#### 1.1.2 内存对齐优化
```cpp
// 确保16字节对齐访问
__global__ void aligned_access_kernel(half *input, half *output, int N) {
  // 使用 half8 (128位) 进行对齐访问
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half8 reg_data = LDST128BITS(input[idx]);  // 128位加载8个half
    // ... 处理数据
    LDST128BITS(output[idx]) = reg_data;       // 128位存储
  }
}
```

#### 1.1.3 分支消除
```cpp
// 使用 predication 而非条件分支
__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
  half pack_x[8], pack_y[8];

  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // 128位加载

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);  // predication
  }
}
```

**性能提升**: 2-3x (相比标量版本)

### 1.2 ⭐️⭐️ Medium 级别 (进阶优化)

**特征**: 引入特定优化策略，显著提升性能

**包含的 kernels**:
- 归一化层: `layer_norm`, `rms_norm`
- Softmax: `softmax`, `online_softmax`
- 点积运算: `dot_product` (优化版本)
- RoPE: 旋转位置编码
- NMS: 非极大值抑制

**核心优化方法**:

#### 1.2.1 分块归约 (Block All Reduce)
```cpp
template <typename T>
__global__ void block_all_reduce_f16x8_pack_kernel(half* input, half* output, int N) {
  extern __shared__ half sdata[];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;

  // 1. Load data with vectorization
  int idx = bid * block_size * 8 + tid * 8;
  half8 reg_input = LDST128BITS(input[idx]);  // 128位加载

  // 2. Thread-level reduction
  half8 partial_sum = reg_input;
  for (int offset = 1; offset < block_size; offset *= 2) {
    __syncthreads();
    int neighbor = tid - offset;
    if (neighbor >= 0) {
      half8 neighbor_data = sdata[neighbor * 8];
      partial_sum = __hadd8(partial_sum, neighbor_data);
    }
  }

  // 3. Store result
  if (tid == 0) {
    LDST128BITS(output[bid * 8]) = partial_sum;
  }
}
```

#### 1.2.2 层归一化优化
```cpp
__global__ void layer_norm_f16x8_pack_kernel(half* input, half* gamma, half* beta,
                                             half* output, int N, int hidden_size) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;

  // 1. 计算均值和方差
  float sum = 0.0f, sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += block_size) {
    float val = __half2float(input[bid * hidden_size + i]);
    sum += val;
    sum_sq += val * val;
  }

  // 2. 归约
  // ... (归约逻辑)

  // 3. 标准化
  float mean = sum / hidden_size;
  float var = sum_sq / hidden_size - mean * mean;
  float inv_std = rsqrtf(var + 1e-5f);

  // 4. 应用 gamma 和 beta
  for (int i = tid; i < hidden_size; i += block_size) {
    float val = (__half2float(input[bid * hidden_size + i]) - mean) * inv_std;
    float gamma_val = __half2float(gamma[i]);
    float beta_val = __half2float(beta[i]);
    output[bid * hidden_size + i] = __float2half(val * gamma_val + beta_val);
  }
}
```

#### 1.2.3 Softmax 在线计算
```cpp
__global__ void online_softmax_f32x4_pack_kernel(float* input, float* output, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int block_size = blockDim.x;

  // 1. 计算最大值 (数值稳定性)
  float max_val = -INFINITY_F;
  for (int i = tid; i < N; i += block_size) {
    max_val = fmaxf(max_val, input[bid * N + i]);
  }

  // 2. 归约最大值
  // ... (归约逻辑)

  // 3. 计算指数和
  float sum_exp = 0.0f;
  for (int i = tid; i < N; i += block_size) {
    float exp_val = expf(input[bid * N + i] - max_val);
    sdata[tid] = exp_val;
    sum_exp += exp_val;
  }

  // 4. 归约求和
  // ... (归约逻辑)

  // 5. 计算最终结果
  for (int i = tid; i < N; i += block_size) {
    float exp_val = expf(input[bid * N + i] - max_val);
    output[bid * N + i] = exp_val / sum_exp;
  }
}
```

**性能提升**: 5-10x (相比基础版本)

### 1.3 ⭐️⭐️⭐️ Hard 级别 (Tensor Core)

**特征**: 使用 Tensor Core 和 MMA 指令，复杂内存管理

**包含的 kernels**:
- SGEMV: 单精度矩阵向量乘法
- HGEVM: 半精度矩阵向量乘法
- SGEMM: 单精度矩阵乘法
- HGEMM: 半精度矩阵乘法 (基础版本)

**核心优化方法**:

#### 1.3.1 Tensor Core MMA 指令使用
```cpp
// MMA 指令宏定义
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
  asm volatile( \
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
    : "=r"(RD0), "=r"(RD1) \
    : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

// HGEMM with Tensor Core
template <const int BM = 128, const int BN = 128, const int BK = 8>
__global__ void hgemm_mma_kernel(half* a, half* b, half* c, int M, int N, int K) {
  // 1. Shared memory tiling
  __shared__ half s_a[BM][BK];
  __shared__ half s_b[BK][BN];

  // 2. Register allocation for MMA
  half frag_a[8];  // 8 registers for A fragment
  half frag_b[4];  // 4 registers for B fragment
  float frag_c[8]; // 8 registers for C fragment (FP32 accumulation)

  // 3. Main computation loop with MMA
  for (int k = 0; k < (K + BK - 1) / BK; ++k) {
    // Load from global memory to shared memory
    // ... (omitted for brevity)

    __syncthreads();

    // Compute with MMA instructions
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      // Load fragments from shared memory
      // ... (omitted for brevity)

      // Execute MMA instruction
      HMMA16816(frag_c[0], frag_c[1], frag_a[0], frag_a[1], frag_a[2], frag_a[3],
                frag_b[0], frag_b[1], frag_c[0], frag_c[1]);
    }

    __syncthreads();
  }

  // 4. Store result
  // ... (omitted for brevity)
}
```

#### 1.3.2 寄存器分块策略
```cpp
template <const int TM = 8, const int TN = 8>
__global__ void register_optimized_kernel(half* A, half* B, half* C, int M, int N, int K) {
  // 寄存器分块
  half reg_a[TM][TN];  // A fragment in registers
  half reg_b[TM][TN];  // B fragment in registers
  float reg_c[TM][TN]; // C fragment in registers (FP32 accumulation)

  // 计算循环
  for (int k = 0; k < K; k += TN) {
    // 加载到寄存器
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        reg_a[i][j] = A[threadIdx.y * TM + i][k + j];
        reg_b[i][j] = B[k + i][threadIdx.x * TN + j];
      }
    }

    // 计算
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        reg_c[i][j] += __hmul(reg_a[i][j], reg_b[i][j]);
      }
    }
  }
}
```

#### 1.3.3 共享内存分块
```cpp
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void hgemm_sliced_k_f16_kernel(half *a, half *b, half *c, int M, int N, int K) {
  // Shared memory tiles
  __shared__ half s_a[BM][BK];
  __shared__ half s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Load to shared memory
  int load_smem_a_m = ty;
  int load_smem_a_k = tx;
  int load_smem_b_k = ty;
  int load_smem_b_n = tx;

  half sum = __float2half(0.f);
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // Load from global to shared
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_m * K + load_gmem_a_k];
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_k * N + load_gmem_b_n];
    __syncthreads();

    // Compute
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      sum += s_a[load_smem_a_m][k] * s_b[k][load_smem_b_n];
    }

    __syncthreads();
  }

  // Store result
  c[store_gmem_c_m * N + store_gmem_c_n] = sum;
}
```

**性能提升**: 10-20x (相比 Medium 版本)

### 1.4 ⭐️⭐️⭐️⭐️ Hard+ 级别 (高级优化)

**特征**: 多阶段流水线，复杂内存优化

**包含的 kernels**:
- FlashAttention-2: 完整实现
- 高级 HGEMM: 多阶段版本
- CUTLASS 集成: 使用 NVIDIA 库

**核心优化方法**:

#### 1.4.1 多阶段流水线
```cpp
template <const int kStage = 2, const int kPad = 8>
__global__ void flash_attn_mma_stages_split_q_kernel(
    half* Q, half* K, half* V, half* O, int QKV_seqlen, int QKV_head) {

  // 1. Multi-stage shared memory
  extern __shared__ half smem[];
  constexpr int Q_tile_size = Br * (kHeadDim + kPad);
  constexpr int KV_tile_size = Bc * (kHeadDim + kPad);
  half* Q_tile_smem = smem;
  half* K_tile_smem = Q_tile_smem + Q_tile_size;
  half* V_tile_smem = K_tile_smem + kStage * KV_tile_size;

  // 2. Asynchronous memory copy
  for (int stage = 0; stage < kStage; ++stage) {
    // Prefetch next tile
    if (stage < kStage - 1) {
      CP_ASYNC_CA(load_smem_K_ptr, load_gmem_K_ptr, bytes);
      CP_ASYNC_CA(load_smem_V_ptr, load_gmem_V_ptr, bytes);
    }

    __syncthreads();

    // Compute with current tile
    // ... (MMA computation)

    // Commit and wait
    CP_ASYNC_COMMIT_GROUP();
    if (stage > 0) {
      CP_ASYNC_WAIT_GROUP(stage - 1);
    }
  }

  // 3. Collective store via warp shuffle
  // ... (omitted for brevity)
}
```

#### 1.4.2 异步内存拷贝
```cpp
// cp.async 宏定义
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes) \
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// 使用示例
__global__ void async_copy_kernel(half* gmem_input, half* smem_output, int N) {
  // 异步拷贝到共享内存
  CP_ASYNC_CA(smem_output, gmem_input, N * sizeof(half));

  // 提交组
  CP_ASYNC_COMMIT_GROUP();

  // 等待所有异步操作完成
  CP_ASYNC_WAIT_ALL();
}
```

#### 1.4.3 寄存器双缓冲
```cpp
template <const int kStage = 2>
__global__ void double_buffer_kernel(half* input, half* output, int N) {
  extern __shared__ half smem[];

  // 双缓冲寄存器
  half reg_buffer[2][BLOCK_SIZE / 32]; // 每个warp一个缓冲

  for (int i = 0; i < N; i += BLOCK_SIZE) {
    int stage = (i / BLOCK_SIZE) % kStage;

    // 异步加载到寄存器
    if (i + BLOCK_SIZE < N) {
      CP_ASYNC_CA(&smem[(stage + 1) % kStage * BLOCK_SIZE],
                  &input[i + BLOCK_SIZE],
                  BLOCK_SIZE * sizeof(half));
    }

    // 从寄存器计算
    compute_with_registers(reg_buffer[stage]);

    // 切换缓冲区
    stage = (stage + 1) % kStage;
  }
}
```

**性能提升**: 20-50x (相比 Hard 版本)

### 1.5 ⭐️⭐️⭐️⭐️⭐️ Hard++ 级别 (前沿技术)

**特征**: 最新优化技术，针对特定硬件优化

**包含的 kernels**:
- FFPA Attention: 更快的 Flash Prefill Attention
- 高级 Swizzle: 手动内存交错优化
- 混合精度: FP16/FP32 混合计算

**核心优化方法**:

#### 1.5.1 FFPA Attention (O(1) SRAM 复杂度)
```cpp
// Fine-grained tiling for constant SRAM usage
template <const int kMmaAtomK = 16>
__global__ void ffpa_tiling_qkv_kernel(half* Q, half* K, half* V, half* O,
                                      int QKV_seqlen, int QKV_head) {
  // SRAM complexity: O(16 * kMmaAtomK) = O(256) constant
  constexpr int SRAM_Q = 16 * kMmaAtomK;    // 256 elements
  constexpr int SRAM_KV = 16 * kMmaAtomK;   // 256 elements

  extern __shared__ half smem[];
  half* s_Q = smem;
  half* s_K = s_Q + SRAM_Q;
  half* s_V = s_K + SRAM_KV;

  // Fine-grained tiling across MMA level
  // ... (omitted for brevity)
}
```

#### 1.5.2 高级 Swizzle (Bank Conflict Free)
```cpp
// Swizzle 函数：避免 bank conflicts
__device__ __host__ __forceinline__ int swizzle_j(int i, int j) {
  return ((int(j / 8) ^ int(i / 4)) % 2) * 8;
}

// 应用示例
__global__ void swizzle_kernel(half* A, half* B, half* C, int M, int N, int K) {
  // ... 计算逻辑
  int smem_addr = i * (N + 8) + swizzle_j(i, j);  // 应用 swizzle
  sdata[smem_addr] = value;
}

// 手动 swizzle 实现
__global__ void manual_swizzle_kernel(half* input, half* output, int M, int N) {
  extern __shared__ half sdata[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 手动 swizzle 地址计算
  int swizzle_offset = ((tx / 8) ^ (ty / 4)) % 2 * 8;
  int smem_addr = ty * (N + 8) + tx + swizzle_offset;

  sdata[smem_addr] = input[threadIdx.y * N + threadIdx.x];
  __syncthreads();

  // 读取时也需要相同的 swizzle
  output[threadIdx.y * N + threadIdx.x] = sdata[smem_addr];
}
```

#### 1.5.3 混合精度计算
```cpp
// 混合精度 FFPA
__global__ void mixed_precision_ffpa_kernel(half* Q, half* K, half* V, half* O,
                                           int QKV_seqlen, int QKV_head) {
  // QK 使用 FP32 精度，PV 使用 FP16 精度
  float reg_qk[16];  // FP32 for QK computation
  half reg_pv[16];   // FP16 for PV computation

  // QK 计算 (FP32)
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float q_val = __half2float(Q[i]);
    float k_val = __half2float(K[i]);
    reg_qk[i] = q_val * k_val;  // FP32 multiplication
  }

  // PV 计算 (FP16)
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    half p_val = __float2half(reg_qk[i]);  // Convert to FP16
    half v_val = V[i];
    reg_pv[i] = __hmul(p_val, v_val);      // FP16 multiplication
  }
}
```

**性能提升**: 50-100x (相比基础版本)

## 2. 核心优化技术详解

### 2.1 内存访问优化

#### 2.1.1 向量化访问模式
```cpp
// 32位向量化 (half2)
#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value))[0])

// 64位向量化 (float2)
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])

// 128位向量化 (float4/half8)
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// 使用示例
__global__ void vectorized_kernel(half* input, half* output, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half8 reg_data = LDST128BITS(input[idx]);  // 128位加载8个half
  // ... 处理数据
  LDST128BITS(output[idx]) = reg_data;       // 128位存储
}
```

#### 2.1.2 内存合并访问
```cpp
// 确保同warp内线程访问连续内存
__global__ void coalesced_access_kernel(float* input, float* output, int N) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int base_addr = blockIdx.x * blockDim.x * 4;  // 4字节对齐

  // 同warp内线程访问连续地址
  float4 data = reinterpret_cast<float4*>(input + base_addr)[lane_id];

  // 处理数据
  // ... (计算逻辑)

  // 存储结果
  reinterpret_cast<float4*>(output + base_addr)[lane_id] = data;
}
```

#### 2.1.3 银行冲突避免
```cpp
// Bank conflict free shared memory layout
template <const int kBankWidth = 32>
__global__ void bank_conflict_free_kernel(half* input, half* output, int M, int N) {
  extern __shared__ half sdata[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 添加 padding 避免 bank conflicts
  constexpr int kPadding = 8;  // 8个half的padding
  int smem_addr = ty * (N + kPadding) + tx;

  // Load with padding
  sdata[smem_addr] = input[ty * N + tx];
  __syncthreads();

  // Process data
  // ... (计算逻辑)

  // Store result
  output[ty * N + tx] = sdata[smem_addr];
}
```

### 2.2 计算优化

#### 2.2.1 Tensor Core 使用
```cpp
// m16n8k16 MMA 指令
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
  asm volatile( \
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
    : "=r"(RD0), "=r"(RD1) \
    : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

// 使用示例
__global__ void mma_compute_kernel(half* A, half* B, half* C, int M, int N, int K) {
  // 准备寄存器
  half RA[4], RB[2], RC[2];
  float RD[2];

  // 执行 MMA
  HMMA16816(RD[0], RD[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
}
```

#### 2.2.2 寄存器分配优化
```cpp
// 寄存器分块策略
template <const int TM = 16, const int TN = 8>
__global__ void register_tiling_kernel(half* A, half* B, half* C, int M, int N, int K) {
  // 寄存器分块
  half reg_a[TM][TN];
  half reg_b[TM][TN];
  float reg_c[TM][TN];

  // 计算循环
  for (int k = 0; k < K; k += TN) {
    // 加载到寄存器
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        reg_a[i][j] = A[threadIdx.y * TM + i][k + j];
        reg_b[i][j] = B[k + i][threadIdx.x * TN + j];
      }
    }

    // 计算
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        reg_c[i][j] += __hmul(reg_a[i][j], reg_b[i][j]);
      }
    }
  }
}
```

#### 2.2.3 循环展开和 unroll
```cpp
// 循环展开优化
template <const int UNROLL_FACTOR = 4>
__global__ void unrolled_kernel(half* input, half* output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 手动循环展开
  #pragma unroll UNROLL_FACTOR
  for (int i = tid; i < N; i += blockDim.x * gridDim.x * UNROLL_FACTOR) {
    // 第一次迭代
    output[i] = __hadd(input[i], __float2half(1.0f));

    // 第二次迭代
    if (i + blockDim.x * gridDim.x < N) {
      output[i + blockDim.x * gridDim.x] =
        __hadd(input[i + blockDim.x * gridDim.x], __float2half(1.0f));
    }

    // 第三次迭代
    if (i + 2 * blockDim.x * gridDim.x < N) {
      output[i + 2 * blockDim.x * gridDim.x] =
        __hadd(input[i + 2 * blockDim.x * gridDim.x], __float2half(1.0f));
    }

    // 第四次迭代
    if (i + 3 * blockDim.x * gridDim.x < N) {
      output[i + 3 * blockDim.x * gridDim.x] =
        __hadd(input[i + 3 * blockDim.x * gridDim.x], __float2half(1.0f));
    }
  }
}
```

### 2.3 流水线优化

#### 2.3.1 多阶段流水线
```cpp
template <const int kStage = 3>
__global__ void pipeline_kernel(half* input, half* output, int N) {
  extern __shared__ half smem[];

  // 阶段缓冲区
  constexpr int BUFFER_SIZE = BLOCK_SIZE;
  half* buffers[kStage];

  for (int s = 0; s < kStage; ++s) {
    buffers[s] = smem + s * BUFFER_SIZE;
  }

  int stage = 0;

  // 主循环
  for (int i = 0; i < N; i += kStage * BLOCK_SIZE) {
    // 1. 预取下一阶段数据
    if (i + (stage + 1) * BLOCK_SIZE < N) {
      CP_ASYNC_CA(buffers[(stage + 1) % kStage],
                  &input[i + (stage + 1) * BLOCK_SIZE],
                  BLOCK_SIZE * sizeof(half));
    }

    // 2. 计算当前阶段
    __syncthreads();
    compute_stage(buffers[stage], &output[i]);

    // 3. 同步和切换阶段
    CP_ASYNC_COMMIT_GROUP();
    if (i > 0) {
      CP_ASYNC_WAIT_GROUP((stage - 1 + kStage) % kStage);
    }

    stage = (stage + 1) % kStage;
  }
}
```

#### 2.3.2 异步内存拷贝
```cpp
// 异步拷贝宏
#define CP_ASYNC_CA(dst, src, bytes) \
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// 使用示例
__global__ void async_pipeline_kernel(half* gmem_input, half* smem_output, int N) {
  // 异步拷贝多个块
  for (int i = 0; i < N; i += BLOCK_SIZE * 2) {
    // 发起多个异步拷贝
    CP_ASYNC_CA(&smem_output[0], &gmem_input[i], BLOCK_SIZE * sizeof(half));
    CP_ASYNC_CA(&smem_output[BLOCK_SIZE], &gmem_input[i + BLOCK_SIZE], BLOCK_SIZE * sizeof(half));

    // 提交组
    CP_ASYNC_COMMIT_GROUP();

    // 计算第一个块
    compute_block(&smem_output[0]);

    // 等待第二个块完成
    CP_ASYNC_WAIT_GROUP(0);

    // 计算第二个块
    compute_block(&smem_output[BLOCK_SIZE]);
  }
}
```

#### 2.3.3 寄存器双缓冲
```cpp
template <const int kStage = 2>
__global__ void double_buffer_kernel(half* input, half* output, int N) {
  // 双缓冲寄存器
  half reg_buffer[2][BLOCK_SIZE / 32]; // 每个warp一个缓冲

  for (int i = 0; i < N; i += BLOCK_SIZE) {
    int current_stage = (i / BLOCK_SIZE) % kStage;
    int next_stage = (current_stage + 1) % kStage;

    // 异步加载到寄存器
    if (i + BLOCK_SIZE < N) {
      CP_ASYNC_CA(®_buffer[next_stage], &input[i + BLOCK_SIZE], BLOCK_SIZE * sizeof(half));
    }

    // 从寄存器计算
    compute_with_registers(reg_buffer[current_stage]);

    // 切换缓冲区
    current_stage = next_stage;
  }
}
```

## 3. 性能优化通用模式

### 3.1 内存层次优化策略

#### 3.1.1 寄存器优先原则
```cpp
// 最大化寄存器使用
template <const int kRegisterCount = 128>
__global__ void register_heavy_kernel(half* input, half* output, int N) {
  // 使用大量寄存器减少内存访问
  half reg_data[kRegisterCount];

  // 预加载到寄存器
  #pragma unroll
  for (int i = 0; i < kRegisterCount; ++i) {
    int idx = blockIdx.x * blockDim.x * kRegisterCount + threadIdx.x * kRegisterCount + i;
    if (idx < N) {
      reg_data[i] = input[idx];
    }
  }

  // 在寄存器中计算
  #pragma unroll
  for (int i = 0; i < kRegisterCount; ++i) {
    reg_data[i] = __hadd(reg_data[i], __float2half(1.0f));
  }

  // 存储结果
  #pragma unroll
  for (int i = 0; i < kRegisterCount; ++i) {
    int idx = blockIdx.x * blockDim.x * kRegisterCount + threadIdx.x * kRegisterCount + i;
    if (idx < N) {
      output[idx] = reg_data[i];
    }
  }
}
```

#### 3.1.2 共享内存优化
```cpp
// 共享内存最佳实践
template <const int TILE_SIZE = 32>
__global__ void shared_memory_optimized_kernel(half* input, half* output, int M, int N) {
  extern __shared__ half sdata[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 1. 加载数据到共享内存 (考虑 bank conflicts)
  int smem_addr = ty * (TILE_SIZE + 8) + tx;  // +8 padding
  sdata[smem_addr] = input[ty * N + tx];

  __syncthreads();

  // 2. 计算 (最大化共享内存重用)
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    half val = sdata[smem_addr];
    // ... 计算逻辑
    sdata[smem_addr] = val;
  }

  __syncthreads();

  // 3. 存储结果
  output[ty * N + tx] = sdata[smem_addr];
}
```

#### 3.1.3 全局内存优化
```cpp
// 全局内存访问优化
__global__ void global_memory_optimized_kernel(half* input, half* output, int N) {
  // 1. 确保内存对齐
  int aligned_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;  // 8个half对齐

  // 2. 向量化访问
  if (aligned_idx < N) {
    half8 data = LDST128BITS(input[aligned_idx]);

    // 3. 计算 (使用向量化操作)
    half8 result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      result[i] = __hadd(data[i], __float2half(1.0f));
    }

    // 4. 存储结果
    LDST128BITS(output[aligned_idx]) = result;
  }
}
```

### 3.2 并行优化策略

#### 3.2.1 Warp 利用优化
```cpp
// 确保 warp 内所有线程都有工作
__global__ void warp_optimized_kernel(half* input, half* output, int N) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int total_warps = (blockDim.x * gridDim.x) / 32;

  // 每个 warp 处理固定数量的元素
  int elements_per_warp = (N + total_warps - 1) / total_warps;
  int warp_start = warp_id * elements_per_warp;
  int warp_end = min(warp_start + elements_per_warp, N);

  // warp 内负载均衡
  for (int i = warp_start + lane_id; i < warp_end; i += 32) {
    output[i] = __hadd(input[i], __float2half(1.0f));
  }
}
```

#### 3.2.2 Occupancy 优化
```cpp
// 使用 __launch_bounds__ 控制资源使用
template <const int MAX_THREADS = 256, const int MIN_BLOCKS = 8>
__global__ void __launch_bounds__(MAX_THREADS, MIN_BLOCKS)
occupancy_optimized_kernel(half* input, half* output, int N) {
  // 减少寄存器使用以提高 occupancy
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 使用更少的寄存器变量
  half temp;

  // 计算 (避免使用大量中间变量)
  for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
    temp = input[i];
    temp = __hadd(temp, __float2half(0.5f));
    output[i] = temp;
  }
}
```

#### 3.2.3 内存合并访问
```cpp
// 确保内存访问合并
__global__ void coalesced_access_kernel(float* input, float* output, int N) {
  int warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  // 每个 warp 处理连续的内存块
  int base_addr = warp_id * 32 * 4;  // 4个float对齐

  // 同warp内线程访问连续地址
  float4 data = reinterpret_cast<float4*>(input + base_addr)[lane_id];

  // 处理数据
  data.x += 1.0f;
  data.y += 1.0f;
  data.z += 1.0f;
  data.w += 1.0f;

  // 存储结果
  reinterpret_cast<float4*>(output + base_addr)[lane_id] = data;
}
```

### 3.3 计算优化策略

#### 3.3.1 指令级并行
```cpp
// 通过流水线提高指令级并行度
__global__ void instruction_parallel_kernel(half* input, half* output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 创建指令级并行
  half reg_a, reg_b, reg_c, reg_d;

  for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
    // 加载阶段
    reg_a = input[i];
    reg_b = input[i + 1];
    reg_c = input[i + 2];
    reg_d = input[i + 3];

    // 计算阶段 (可以并行执行)
    reg_a = __hadd(reg_a, __float2half(1.0f));
    reg_b = __hadd(reg_b, __float2half(2.0f));
    reg_c = __hadd(reg_c, __float2half(3.0f));
    reg_d = __hadd(reg_d, __float2half(4.0f));

    // 存储阶段
    output[i] = reg_a;
    output[i + 1] = reg_b;
    output[i + 2] = reg_c;
    output[i + 3] = reg_d;
  }
}
```

#### 3.3.2 数学函数优化
```cpp
// 使用硬件加速的数学函数
__global__ void math_optimized_kernel(float* input, float* output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N) {
    float val = input[tid];

    // 使用硬件加速函数
    float result = __fadd_rn(val, 1.0f);        // 硬件加法
    result = __fmul_rn(result, 2.0f);           // 硬件乘法
    result = __frcp_rn(result);                 // 硬件倒数
    result = __fsqrt_rn(result);                // 硬件平方根
    result = __fexpf(result);                   // 硬件指数函数

    output[tid] = result;
  }
}
```

#### 3.3.3 精度选择优化
```cpp
// 根据精度需求选择合适的数据类型
template <typename T>
__global__ void precision_optimized_kernel(T* input, T* output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N) {
    T val = input[tid];

    // 根据数据类型选择合适的计算
    if constexpr (std::is_same_v<T, half>) {
      // FP16: 使用 half 精度计算
      output[tid] = __hadd(val, __float2half(1.0f));
    } else if constexpr (std::is_same_v<T, float>) {
      // FP32: 使用 float 精度计算
      output[tid] = val + 1.0f;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      // BF16: 使用 bfloat16 精度计算
      output[tid] = __hadd(val, __float2bfloat16(1.0f));
    }
  }
}
```

## 4. 关键实现思路总结

### 4.1 FlashAttention 2.0 核心思想

#### 4.1.1 分块计算策略
```cpp
// FlashAttention-2 分块计算
template <const int Br = 64, const int Bc = 64>
__global__ void flash_attention_2_kernel(half* Q, half* K, half* V, half* O,
                                        int seqlen, int head_dim) {
  // 1. 将 Q 分块为 [Br, d] 大小
  // 2. 将 K,V 分块为 [Bc, d] 大小
  // 3. 计算 Q@K^T -> P[Br, Bc]
  // 4. 计算 P@V -> O[Br, d]

  // 分块循环
  for (int q_tile = 0; q_tile < (seqlen + Br - 1) / Br; ++q_tile) {
    for (int kv_tile = 0; kv_tile < (seqlen + Bc - 1) / Bc; ++kv_tile) {
      // 加载 Q, K, V 分块
      load_tile(Q, q_tile, Br, head_dim);
      load_tile(K, kv_tile, Bc, head_dim);
      load_tile(V, kv_tile, Bc, head_dim);

      // 计算 Q@K^T
      compute_qk();

      // Softmax
      apply_softmax();

      // 计算 P@V
      compute_pv();

      // 累积结果
      accumulate_result();
    }
  }
}
```

#### 4.1.2 在线计算策略
```cpp
// 在线 Softmax 计算
template <const int Br = 64, const int Bc = 64>
__global__ void online_softmax_kernel(half* P, half* O, int seqlen, int head_dim) {
  // 在线计算 softmax，避免存储完整的 P 矩阵

  for (int kv_tile = 0; kv_tile < (seqlen + Bc - 1) / Bc; ++kv_tile) {
    // 1. 计算当前 tile 的最大值
    float max_val = compute_tile_max(P);

    // 2. 计算指数和
    float sum_exp = compute_tile_exp_sum(P, max_val);

    // 3. 应用 softmax
    apply_tile_softmax(P, max_val, sum_exp);

    // 4. 累积到输出
    accumulate_tile_to_output(P, O);
  }
}
```

### 4.2 HGEMM 优化技巧

#### 4.2.1 Tiling 策略
```cpp
// HGEMM Tiling 策略
template <const int BM = 128, const int BN = 128, const int BK = 32>
__global__ void hgemm_tiling_kernel(half* A, half* B, half* C, int M, int N, int K) {
  // 1. 将大矩阵分块为小块
  // A: [M,K] -> [BM,BK] x (M/BM, K/BK)
  // B: [K,N] -> [BK,BN] x (K/BK, N/BN)
  // C: [M,N] -> [BM,BN] x (M/BM, N/BN)

  // 2. 循环分块
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    for (int bn = 0; bn < (N + BN - 1) / BN; ++bn) {
      for (int bm = 0; bm < (M + BM - 1) / BM; ++bm) {
        // 加载分块到共享内存
        load_block_to_shared(A, bm, bk, BM, BK);
        load_block_to_shared(B, bk, bn, BK, BN);

        __syncthreads();

        // 计算分块乘法
        compute_block_multiply(C, bm, bn);

        __syncthreads();
      }
    }
  }
}
```

#### 4.2.2 寄存器重用
```cpp
// 寄存器数据重用策略
template <const int TM = 8, const int TN = 8, const int TK = 16>
__global__ void register_reuse_kernel(half* A, half* B, half* C, int M, int N, int K) {
  // 寄存器分块
  half reg_a[TM][TK];
  half reg_b[TK][TN];
  float reg_c[TM][TN];

  // 主循环
  for (int k = 0; k < K; k += TK) {
    // 加载到寄存器 (重用寄存器)
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TK; ++j) {
        reg_a[i][j] = A[threadIdx.y * TM + i][k + j];
      }
    }

    #pragma unroll
    for (int i = 0; i < TK; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        reg_b[i][j] = B[k + i][threadIdx.x * TN + j];
      }
    }

    // 计算 (最大化寄存器重用)
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
          reg_c[i][j] += __hmul(reg_a[i][kk], reg_b[kk][j]);
        }
      }
    }
  }

  // 存储结果
  // ... (omitted for brevity)
}
```

### 4.3 通用优化模式

#### 4.3.1 Launch Bounds 使用
```cpp
// 使用 __launch_bounds__ 控制资源使用
template <const int MAX_THREADS = 512, const int MIN_BLOCKS = 4>
__global__ void __launch_bounds__(MAX_THREADS, MIN_BLOCKS)
launch_bounds_optimized_kernel(half* input, half* output, int N) {
  // 编译器会根据指定的 bounds 优化寄存器使用
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N) {
    // 减少寄存器使用以提高 occupancy
    half val = input[tid];
    val = __hadd(val, __float2half(1.0f));
    output[tid] = val;
  }
}
```

#### 4.3.2 内存预取
```cpp
// 内存预取策略
__global__ void prefetch_kernel(half* input, half* output, int N) {
  // 预取下一块数据
  if (threadIdx.x == 0) {
    __builtin_prefetch(&input[blockIdx.x * BLOCK_SIZE + 1024], 0, 1);
  }

  __syncthreads();

  // 处理当前块
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (idx < N) {
    output[idx] = __hadd(input[idx], __float2half(1.0f));
  }
}
```

#### 4.3.3 分支消除
```cpp
// 使用 predication 消除分支
__global__ void branch_elimination_kernel(half* input, half* output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N) {
    half val = input[tid];
    half threshold = __float2half(0.5f);

    // 使用 predication 而非条件分支
    half result = __hgt(val, threshold) ? val : __float2half(0.0f);

    output[tid] = result;
  }
}
```

## 5. 实际性能数据

### 5.1 不同优化级别的性能对比

| 优化级别 | 代表 kernels | 性能提升 | 关键技术 |
|---------|-------------|---------|---------|
| Basic | 基础实现 | 1x (基准) | 直接实现 |
| Easy | elementwise, relu | 2-3x | 向量化访问 |
| Medium | layer_norm, softmax | 5-10x | 共享内存, 分块归约 |
| Hard | HGEMM, SGEMM | 10-20x | Tensor Core, 寄存器优化 |
| Hard+ | FlashAttention-2 | 20-50x | 多阶段流水线, 异步拷贝 |
| Hard++ | FFPA Attention | 50-100x | O(1) SRAM, 高级 swizzle |

### 5.2 实际硬件性能数据

#### 5.2.1 HGEMM 性能
- **NVIDIA L20**: 达到 cuBLAS 性能的 98%~100%
- **RTX 4090**: 达到 cuBLAS 性能的 99%~100%
- **RTX 3080 Laptop**: 达到 cuBLAS 性能的 98%~99%

#### 5.2.2 FlashAttention-2 性能
- **小规模 attention** (B≤4, H≤48, SeqLen≤8192, D≤64):
  - 比官方 FA2 快 **1.5x**
  - RTX 3080 Laptop: 55 TFLOPS (D=64)
- **大规模 attention**:
  - 仍有性能差距，正在优化中

#### 5.2.3 FFPA Attention 性能
- **比 SDPA 快 1.8x~3x**
- **O(1) SRAM 复杂度**: 支持大 head_dim (256+)
- **L20**: ~1.9x↑ vs SDPA EA
- **A30**: ~1.8x↑ vs SDPA EA
- **RTX 4090**: ~2.1x↑ vs SDPA EA

### 5.3 优化效果分析

#### 5.3.1 内存带宽利用
- **基础版本**: 30-40% 带宽利用率
- **Easy 优化**: 50-60% 带宽利用率
- **Medium 优化**: 70-80% 带宽利用率
- **Hard 优化**: 85-95% 带宽利用率
- **Hard+ 优化**: 95%+ 带宽利用率

#### 5.3.2 计算单元利用率
- **基础版本**: 20-30% 计算利用率
- **Easy 优化**: 40-50% 计算利用率
- **Medium 优化**: 60-70% 计算利用率
- **Hard 优化**: 80-90% 计算利用率 (Tensor Core)
- **Hard+ 优化**: 90%+ 计算利用率

### 5.4 性能瓶颈分析

#### 5.4.1 内存瓶颈
- **全局内存延迟**: 通过流水线和异步拷贝缓解
- **共享内存 bank conflicts**: 通过 swizzle 消除
- **寄存器压力**: 通过分块和复用缓解

#### 5.4.2 计算瓶颈
- **指令吞吐量**: 通过 Tensor Core 和向量化提升
- **分支预测**: 通过 predication 消除分支
- **数值精度**: 通过混合精度平衡精度和性能

## 6. 总结

LeetCUDA 项目提供了从基础到高级的完整 CUDA kernel 优化学习路径，涵盖了现代 GPU 编程的最佳实践。通过系统性的优化策略，可以实现 2-100x 的性能提升。

### 6.1 优化原则
1. **渐进式优化**: 从简单到复杂，逐步应用优化技术
2. **硬件感知**: 根据 GPU 架构特性选择合适的优化策略
3. **平衡原则**: 在内存、计算、并行度之间找到最佳平衡点
4. **实测验证**: 所有优化都需要通过实际性能测试验证

### 6.2 学习建议
1. **从 Easy 开始**: 先掌握基础的向量化和内存对齐
2. **理解原理**: 深入理解每种优化技术的原理和适用场景
3. **实践验证**: 通过实际代码实现和性能测试巩固理解
4. **持续学习**: 关注最新的 GPU 架构和优化技术

### 6.3 应用场景
- **深度学习**: 神经网络算子优化
- **科学计算**: 矩阵运算、数值模拟
- **图形处理**: 图像滤波、几何变换
- **数据分析**: 大规模数据处理、统计计算

这个优化方法总结为 CUDA 高性能编程提供了全面的参考，帮助开发者理解和实现高效的 GPU 代码。