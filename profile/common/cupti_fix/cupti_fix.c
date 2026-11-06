/*
 * Thor（JetPack 7.2 / CUDA 13.0 / CUPTI 2025.3.1）上 torch.profiler 的部分修复。
 *
 * 现象：torch.profiler 报 "GPU trace is empty!"，连 CUDA runtime API 事件也没有。
 * 原因 1（本文件修复）：Kineto 停止采集时会调用
 *     cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER, 0)
 *   在这个版本的 CUPTI 上，采集后设置该属性会导致已采集的 activity buffer 全部丢失。
 *   用 LD_PRELOAD 拦截并跳过这一属性后，CPU 侧的 CUDA runtime/driver API 事件
 *   （cudaLaunchKernel、cudaStreamSynchronize、cudaMemcpyAsync ...）可以正常采集。
 * 原因 2（无法修复）：CUPTI 返回的 GPU 侧记录（kernel、memset、memcpy）时间戳全部为 0，
 *   被 Kineto 判为 Out-of-range 后丢弃。宿主机上（root、完整权限）同样如此，属于平台限制。
 *   → GPU 侧时间线请使用 nsys（第 04 章）。
 *
 * 编译：gcc -O2 -shared -fPIC -o libcupti_fix.so cupti_fix.c -ldl
 * 使用：LD_PRELOAD=/root/data/learn/profile/common/cupti_fix/libcupti_fix.so python xxx.py
 *       或在脚本开头调用 common.cupti_fix.ensure()（会自动带上 LD_PRELOAD 重新启动当前进程）
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

#define ATTR_PER_THREAD_ACTIVITY_BUFFER 9 /* CUpti_ActivityAttribute 枚举中的数值 */

typedef int (*set_attr_fn)(int, size_t *, void *);

int cuptiActivitySetAttribute(int attr, size_t *size, void *value) {
    static set_attr_fn real = NULL;
    if (attr == ATTR_PER_THREAD_ACTIVITY_BUFFER) return 0; /* CUPTI_SUCCESS：跳过 */
    if (!real) {
        /* 本库通过 LD_PRELOAD 先于 libcupti 加载，RTLD_NEXT 找不到符号，所以显式 dlopen */
        void *h = dlopen("libcupti.so.13", RTLD_NOW | RTLD_GLOBAL);
        real = h ? (set_attr_fn)dlsym(h, "cuptiActivitySetAttribute") : NULL;
        if (!real) return 1; /* CUPTI_ERROR_INVALID_PARAMETER */
    }
    return real(attr, size, value);
}
