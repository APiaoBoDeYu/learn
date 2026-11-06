// 03-3.8 最小 CUPTI activity 测试：kernel / memset / memcpy 的 GPU 时间戳是否为 0？
//   nvcc -O2 -arch=sm_110 cupti_timestamp_test.cu -lcupti -o /tmp/cupti_ts && /tmp/cupti_ts
// 旧容器（v1.0.0）和宿主机 root 上：kernel、memset 记录的 start/end 全部为 0，runtime 记录正常。
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { CUptiResult r = (x); if (r != CUPTI_SUCCESS) { const char* s; cuptiGetResultString(r, &s); \
                    printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s); exit(1); } } while (0)

static int n_kernel = 0, n_kernel_zero = 0;

__global__ void busy(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) for (int k = 0; k < 1000; ++k) x[i] = x[i] * 0.999f + 0.001f;
}

static void CUPTIAPI buf_requested(uint8_t** buf, size_t* size, size_t* max_records) {
    *size = 8 << 20;
    *buf = (uint8_t*)aligned_alloc(8, *size);
    *max_records = 0;
}

static void CUPTIAPI buf_completed(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
    CUpti_Activity* rec = NULL;
    while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
        switch (rec->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                CUpti_ActivityKernel10* k = (CUpti_ActivityKernel10*)rec;
                n_kernel++; if (k->start == 0 || k->end == 0) n_kernel_zero++;
                if (n_kernel <= 3) printf("  KERNEL  %-6s start=%llu end=%llu dur=%lld ns\n", k->name,
                                          (unsigned long long)k->start, (unsigned long long)k->end, (long long)(k->end - k->start));
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                CUpti_ActivityMemset4* m = (CUpti_ActivityMemset4*)rec;
                printf("  MEMSET  start=%llu end=%llu\n", (unsigned long long)m->start, (unsigned long long)m->end);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                CUpti_ActivityMemcpy6* m = (CUpti_ActivityMemcpy6*)rec;
                printf("  MEMCPY  start=%llu end=%llu bytes=%llu\n", (unsigned long long)m->start, (unsigned long long)m->end,
                       (unsigned long long)m->bytes);
                break;
            }
            case CUPTI_ACTIVITY_KIND_RUNTIME: {
                CUpti_ActivityAPI* a = (CUpti_ActivityAPI*)rec;
                static int shown = 0;
                if (shown++ < 2) printf("  RUNTIME cbid=%u start=%llu end=%llu\n", a->cbid, (unsigned long long)a->start,
                                        (unsigned long long)a->end);
                break;
            }
            default: break;
        }
    }
    free(buf);
}

int main() {
    uint32_t ver; CK(cuptiGetVersion(&ver)); printf("CUPTI API version %u\n", ver);
    CK(cuptiActivityRegisterCallbacks(buf_requested, buf_completed));
    CK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    int n = 1 << 20; float* d; float* h = (float*)malloc(n * 4);
    cudaMalloc(&d, n * 4);
    cudaMemset(d, 0, n * 4);
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    for (int i = 0; i < 5; ++i) busy<<<(n + 255) / 256, 256>>>(d, n);
    cudaDeviceSynchronize();
    CK(cuptiActivityFlushAll(1));
    printf("kernel 记录 %d 条，其中时间戳为 0 的 %d 条 → %s\n", n_kernel, n_kernel_zero,
           n_kernel && n_kernel_zero == 0 ? "GPU 时间戳正常" : "GPU 时间戳为 0（平台问题仍然存在）");
    return 0;
}
