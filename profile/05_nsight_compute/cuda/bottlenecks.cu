// 第 05 章：用来在 Nsight Compute 中"制造"各类瓶颈的 kernel。
// 每组都有"问题版"和"修复版"，接口统一为 extern "C" 的启动函数，由 kernels.py 通过 ctypes 调用。
// 编译：kernels.py 自动执行 nvcc -O3 -lineinfo -arch=sm_110 --shared（见 kernels.py:build）
#include <cuda_runtime.h>
#include <stdint.h>

#define EXPORT extern "C"

// ============================================================================
// 1. Memory-bound：逐元素运算
// ============================================================================

// 1a 合并访存：相邻线程访问相邻地址，一个 warp 的 32 个 float 正好是 4 个 32B sector
__global__ void scale_coalesced(const float* __restrict__ x, float* __restrict__ y, float a, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i];
}

// 1b 非合并访存：把数组看成 [stride][n/stride]，按列读取。
//    相邻线程地址相差 (n/stride)*4 字节 → 一个 warp 读 32 个不同的 sector，每个 sector 只用 4/32 字节
__global__ void scale_strided(const float* __restrict__ x, float* __restrict__ y, float a, int64_t n, int stride) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t cols = n / stride;
        int64_t j = (i % stride) * cols + i / stride;   // 第 i 个线程读第 (i%stride) 行、第 (i/stride) 列
        y[j] = a * x[j];
    }
}

// 1c 向量化：每个线程用 float4 一次读写 16 字节，指令数变为 1/4
__global__ void scale_vec4(const float4* __restrict__ x, float4* __restrict__ y, float a, int64_t n4) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = x[i];
        v.x *= a; v.y *= a; v.z *= a; v.w *= a;
        y[i] = v;
    }
}

// 1d 融合：y = relu(a*x + b)，一次读、一次写（PyTorch eager 要 3 个 kernel、3 次读 3 次写）
__global__ void fused_axpb_relu(const float* __restrict__ x, float* __restrict__ y, float a, float b, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(a * x[i] + b, 0.f);
}

EXPORT void launch_scale_coalesced(const float* x, float* y, float a, int64_t n, cudaStream_t s) {
    scale_coalesced<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, y, a, n);
}
EXPORT void launch_scale_strided(const float* x, float* y, float a, int64_t n, int stride, cudaStream_t s) {
    scale_strided<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, y, a, n, stride);
}
EXPORT void launch_scale_vec4(const float* x, float* y, float a, int64_t n, cudaStream_t s) {
    int64_t n4 = n / 4;
    scale_vec4<<<(unsigned)((n4 + 255) / 256), 256, 0, s>>>((const float4*)x, (float4*)y, a, n4);
}
EXPORT void launch_fused_axpb_relu(const float* x, float* y, float a, float b, int64_t n, cudaStream_t s) {
    fused_axpb_relu<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, y, a, b, n);
}

// ============================================================================
// 2. Register-bound：每个线程独立计算一个 8x8 小矩阵乘法 C = A·B
//    （机器人里很常见：批量的位姿变换、卡尔曼滤波的小矩阵运算）
//    数据布局为 SoA：A[e][m]，e = 0..63 是矩阵元素下标，m 是批次下标 → 一个 warp 读同一个 e 时地址连续，访存是合并的
// ============================================================================

template <int D>
__device__ __forceinline__ void bmm_thread(const float* __restrict__ A, const float* __restrict__ B,
                                           float* __restrict__ C, int64_t m, int64_t batch) {
    float a[D * D], b[D * D], c[D * D];          // 3 x 64 个 float，全部放在寄存器里
#pragma unroll
    for (int e = 0; e < D * D; ++e) { a[e] = A[e * batch + m]; b[e] = B[e * batch + m]; }
#pragma unroll
    for (int r = 0; r < D; ++r)
#pragma unroll
        for (int col = 0; col < D; ++col) {
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < D; ++k) acc += a[r * D + k] * b[k * D + col];
            c[r * D + col] = acc;
        }
#pragma unroll
    for (int e = 0; e < D * D; ++e) C[e * batch + m] = c[e];
}

// 2a 问题版：一个线程算一整个矩阵，寄存器用量很大 → 每个 SM 能驻留的 warp 很少
__global__ void bmm8_per_thread(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int64_t batch) {
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (m < batch) bmm_thread<8>(A, B, C, m, batch);
}

// 2b 同样的代码，用 __maxnreg__ 限制每线程最多 64 个寄存器（等价于对这个 kernel 单独使用 -maxrregcount=64）
//    → 编译器只能把放不下的值 spill 到 local memory（实际在 L1/L2/DRAM 上）
__global__ void __maxnreg__(64) bmm8_per_thread_maxreg64(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int64_t batch) {
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (m < batch) bmm_thread<8>(A, B, C, m, batch);
}

// 2c 修复版：改变并行粒度。一个 block = 32 个矩阵 × 8 行；warp w 负责 32 个矩阵的第 w 行。
//    每个线程只需要 A 的一行（8 个）+ 8 个累加器；B 由同一 block 的 8 个 warp 共同读取，第 2~8 次读取命中 L1
__global__ void bmm8_row_per_thread(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int64_t batch) {
    const int D = 8;
    int lane = threadIdx.x & 31, r = threadIdx.x >> 5;           // blockDim = 256 = 8 个 warp
    int64_t m = (int64_t)blockIdx.x * 32 + lane;
    if (m >= batch) return;
    float a[D], acc[D];
#pragma unroll
    for (int k = 0; k < D; ++k) { a[k] = A[(r * D + k) * batch + m]; acc[k] = 0.f; }
#pragma unroll
    for (int k = 0; k < D; ++k)
#pragma unroll
        for (int col = 0; col < D; ++col) acc[col] += a[k] * B[(k * D + col) * batch + m];
#pragma unroll
    for (int col = 0; col < D; ++col) C[(r * D + col) * batch + m] = acc[col];
}

EXPORT void launch_bmm8(int variant, const float* A, const float* B, float* C, int64_t batch, int block, cudaStream_t s) {
    if (variant == 0) bmm8_per_thread<<<(unsigned)((batch + block - 1) / block), block, 0, s>>>(A, B, C, batch);
    else if (variant == 1) bmm8_per_thread_maxreg64<<<(unsigned)((batch + block - 1) / block), block, 0, s>>>(A, B, C, batch);
    else bmm8_row_per_thread<<<(unsigned)((batch + 31) / 32), 256, 0, s>>>(A, B, C, batch);
}

// ---------------------------------------------------------------------------
// 2'. 低 ILP 的情况：每个元素维护 S 个状态（例如一组不同衰减系数的滤波器），沿时间维逐步更新
//     s[j] = s[j]*w_j + x_t。每轮循环只有 1 个 load，后面的 S 个 FMA 都依赖它 → 只能靠"别的 warp"来隐藏 load 延迟
//     x: [T][n]，out: [S][n]
// ---------------------------------------------------------------------------
template <int S, int J0, int PF>   // J0 是编译期常量：系数 0.5+0.003*(J0+j) 直接编码成指令里的立即数，不占寄存器
__device__ __forceinline__ void filter_bank(const float* __restrict__ x, float* __restrict__ out, int T, int64_t n,
                                            int64_t m) {
    float s[S];
#pragma unroll
    for (int j = 0; j < S; ++j) s[j] = 0.f;
    if (PF == 0) {
        // 朴素写法：load 之后立刻要用 → 每轮都要等一次完整的访存延迟（约 800+ 周期）
#pragma unroll 1
        for (int t = 0; t < T; ++t) {
            float v = x[(int64_t)t * n + m];
#pragma unroll
            for (int j = 0; j < S; ++j) s[j] = fmaf(s[j], 0.5f + 0.003f * (J0 + j), v);
        }
    } else {
        // 软件预取：提前 PF 轮发出 load，用当前这轮的计算掩盖下一轮的访存延迟。只多用 PF 个寄存器
        float buf[PF > 0 ? PF : 1];
#pragma unroll
        for (int k = 0; k < PF; ++k) buf[k] = k < T ? x[(int64_t)k * n + m] : 0.f;
#pragma unroll 1
        for (int t = 0; t < T; t += PF) {
#pragma unroll
            for (int k = 0; k < PF; ++k) {
                float v = buf[k];
                int tn = t + PF + k;
                buf[k] = tn < T ? x[(int64_t)tn * n + m] : 0.f;
                if (t + k < T) {
#pragma unroll
                    for (int j = 0; j < S; ++j) s[j] = fmaf(s[j], 0.5f + 0.003f * (J0 + j), v);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < S; ++j) out[(int64_t)(J0 + j) * n + m] = s[j];
}

// 问题版：一个线程负责 128 个状态 → 寄存器多（168 个）、Occupancy 只有 25%
template <int PF>
__global__ void filter128_per_thread(const float* __restrict__ x, float* __restrict__ out, int T, int64_t n) {
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (m < n) filter_bank<128, 0, PF>(x, out, T, n, m);
}
// "提高 Occupancy"：4 个线程分担一个元素的 128 个状态（每个线程 32 个）。4 个线程读同一个 x（同一地址，广播），
//    总 FMA 数不变，每个线程的寄存器少了，SM 上能驻留的 warp 多了；但每个 warp 两次 load 之间的计算也只有原来的 1/4
//    线程映射：block 内 threadIdx.x = q*64 + l（q = 0..3 是状态分组），保证同一 warp 内的 x 访问仍然合并
template <int PF>
__global__ void filter128_split4(const float* __restrict__ x, float* __restrict__ out, int T, int64_t n) {
    int q = threadIdx.x / 64, l = threadIdx.x % 64;            // blockDim = 256
    int64_t m = (int64_t)blockIdx.x * 64 + l;
    if (m >= n) return;
    switch (q) {   // q 在 warp 内是一致的（每 64 个线程一组），不会产生 divergence
        case 0: filter_bank<32, 0, PF>(x, out, T, n, m); break;
        case 1: filter_bank<32, 32, PF>(x, out, T, n, m); break;
        case 2: filter_bank<32, 64, PF>(x, out, T, n, m); break;
        default: filter_bank<32, 96, PF>(x, out, T, n, m); break;
    }
}

EXPORT void launch_filter(int variant, const float* x, float* out, int T, int64_t n, cudaStream_t s) {
    switch (variant) {   // 0=每线程 128 个状态 1=4 个线程分担 2=0+预取 3=1+预取
        case 0: filter128_per_thread<0><<<(unsigned)((n + 127) / 128), 128, 0, s>>>(x, out, T, n); break;
        case 1: filter128_split4<0><<<(unsigned)((n + 63) / 64), 256, 0, s>>>(x, out, T, n); break;
        case 2: filter128_per_thread<4><<<(unsigned)((n + 127) / 128), 128, 0, s>>>(x, out, T, n); break;
        default: filter128_split4<4><<<(unsigned)((n + 63) / 64), 256, 0, s>>>(x, out, T, n); break;
    }
}

// ============================================================================
// 3. Control-flow-bound：warp divergence
// ============================================================================

// 两个分支的 FMA 数量相同（都是 iters 次），但循环结构不同，
// 避免编译器把 if/else 转换成"先选常数、再跑同一个循环"的无分支代码（那样就看不到 divergence 了）
__device__ __forceinline__ float path_a(float v, int iters) {
    for (int k = 0; k < iters; ++k) v = fmaf(v, 0.9999f, 0.0001f);
    return v;
}
__device__ __forceinline__ float path_b(float v, int iters) {
    for (int k = 0; k < iters / 2; ++k) { v = fmaf(v, 1.0001f, -0.0001f); v = fmaf(v, 0.9998f, 0.0002f); }
    return v;
}

// 3a 问题版：按线程奇偶分支 → 每个 warp 内一半线程走 A、一半走 B，两条路径串行执行
__global__ void branch_divergent(float* __restrict__ x, int64_t n, int iters) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (threadIdx.x & 1) v = path_a(v, iters);
    else                 v = path_b(v, iters);
    x[i] = v;
}

// 3b 修复版：按 warp 分支（threadIdx.x / 32 的奇偶）→ 每个 warp 内所有线程走同一条路径。总工作量完全相同
__global__ void branch_warp_uniform(float* __restrict__ x, int64_t n, int iters) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if ((threadIdx.x >> 5) & 1) v = path_a(v, iters);
    else                        v = path_b(v, iters);
    x[i] = v;
}

// 3c 数据相关的循环次数：每个元素的迭代次数 cnt[i] 不同，warp 的耗时 = 32 个线程中的最大值
//    修复方法不改 kernel，而是在调用前把元素按 cnt 排序（见 04_control_flow.py）
__global__ void loop_varcount(float* __restrict__ x, const int* __restrict__ cnt, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    int c = cnt[i];
    for (int k = 0; k < c; ++k) v = v * 0.9999f + 0.0001f;
    x[i] = v;
}

EXPORT void launch_branch(int variant, float* x, int64_t n, int iters, cudaStream_t s) {
    unsigned g = (unsigned)((n + 255) / 256);
    if (variant == 0) branch_divergent<<<g, 256, 0, s>>>(x, n, iters);
    else              branch_warp_uniform<<<g, 256, 0, s>>>(x, n, iters);
}
EXPORT void launch_loop_varcount(float* x, const int* cnt, int64_t n, cudaStream_t s) {
    loop_varcount<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(x, cnt, n);
}

// ============================================================================
// 4. Shared memory bank conflict：矩阵转置 out[c][r] = in[r][c]（rows x cols，row-major）
//    block = (32, 8)，每个 block 处理 32x32 的 tile，每个线程处理 4 行
// ============================================================================
#define TILE 32
#define BROWS 8

// 4a 朴素版：读合并，写不合并（相邻线程写的地址相差 rows*4 字节）
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    int c = blockIdx.x * TILE + threadIdx.x;
    int r0 = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        out[(int64_t)c * rows + (r0 + j)] = in[(int64_t)(r0 + j) * cols + c];
}

// 4b shared memory 版：先合并地读进 tile，再按列从 tile 中取出、合并地写出去。
//    问题：tile[threadIdx.x][...] 按列读时，相邻线程的地址相差 32 个 float = 128 字节 → 全部落在同一个 bank → 32-way conflict
__global__ void transpose_smem(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[TILE][TILE];
    int c = blockIdx.x * TILE + threadIdx.x, r = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) tile[threadIdx.y + j][threadIdx.x] = in[(int64_t)(r + j) * cols + c];
    __syncthreads();
    c = blockIdx.y * TILE + threadIdx.x; r = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) out[(int64_t)(r + j) * rows + c] = tile[threadIdx.x][threadIdx.y + j];
}

// 4c 修复 1：padding。每行多放 1 个 float，相邻行错开一个 bank：(x*33 + y) % 32 = (x + y) % 32，32 个线程落在 32 个不同的 bank
__global__ void transpose_smem_pad(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];
    int c = blockIdx.x * TILE + threadIdx.x, r = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) tile[threadIdx.y + j][threadIdx.x] = in[(int64_t)(r + j) * cols + c];
    __syncthreads();
    c = blockIdx.y * TILE + threadIdx.x; r = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) out[(int64_t)(r + j) * rows + c] = tile[threadIdx.x][threadIdx.y + j];
}

// 4d 修复 2：XOR swizzle。不浪费 shared memory：第 row 行的第 col 列存放在 col ^ row 的位置
__global__ void transpose_smem_swizzle(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[TILE][TILE];
    int c = blockIdx.x * TILE + threadIdx.x, r = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) {
        int row = threadIdx.y + j;
        tile[row][threadIdx.x ^ row] = in[(int64_t)(r + j) * cols + c];
    }
    __syncthreads();
    c = blockIdx.y * TILE + threadIdx.x; r = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) {
        int row = threadIdx.x, col = threadIdx.y + j;
        out[(int64_t)(r + j) * rows + c] = tile[row][col ^ row];
    }
}

// 4e 参照：纯拷贝（读写都合并，没有转置）→ 转置能达到的上限
__global__ void copy_tile(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    int c = blockIdx.x * TILE + threadIdx.x, r = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) out[(int64_t)(r + j) * cols + c] = in[(int64_t)(r + j) * cols + c];
}

EXPORT void launch_transpose(int variant, const float* in, float* out, int rows, int cols, cudaStream_t s) {
    dim3 grid(cols / TILE, rows / TILE), block(TILE, BROWS);
    switch (variant) {
        case 0: transpose_naive<<<grid, block, 0, s>>>(in, out, rows, cols); break;
        case 1: transpose_smem<<<grid, block, 0, s>>>(in, out, rows, cols); break;
        case 2: transpose_smem_pad<<<grid, block, 0, s>>>(in, out, rows, cols); break;
        case 3: transpose_smem_swizzle<<<grid, block, 0, s>>>(in, out, rows, cols); break;
        default: copy_tile<<<grid, block, 0, s>>>(in, out, rows, cols); break;
    }
}
