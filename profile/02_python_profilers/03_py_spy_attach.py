"""02-03b 在 privileged 容器（或有 CAP_SYS_PTRACE）中，py-spy 直接 attach 一个"不是自己子进程"的 Python 进程。

    /opt/venv/bin/python 02_python_profilers/03_py_spy_attach.py   → outputs/py_spy_dump_privileged.log

目标进程用 start_new_session=True 启动，并且**不加** --allow-ptrace（不调用 prctl）。
在旧容器（docker 默认权限，ptrace_scope=1）中这样做会报 Permission denied，见 README 2.4。
"""
import subprocess
import sys
import time
# 目标进程：独立会话启动，不是 py-spy 的子进程，也没有调用 prctl（不加 --allow-ptrace）
p = subprocess.Popen([sys.executable, "02_python_profilers/workload.py", "--steps", "3000"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
try:
    for _ in range(60):
        time.sleep(0.5)
        r = subprocess.run(["py-spy", "dump", "--pid", str(p.pid)], capture_output=True, text=True)
        if "pipeline.py" in r.stdout:
            break
    print("returncode", r.returncode, r.stderr.strip()[:200])
    print("\n".join(r.stdout.splitlines()[:12]))

finally:
    p.kill()
