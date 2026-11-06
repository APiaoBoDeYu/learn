"""模拟一个长期运行的服务。HANDLE_SIGTERM=1 时注册 SIGTERM 处理函数（模拟"优雅关闭"）。"""
import os
import signal
import sys
import time


def on_sigterm(signum, frame):
    print(f"[app pid={os.getpid()}] got SIGTERM, cleaning up...", flush=True)
    time.sleep(0.5)  # 模拟清理：关相机、断开机械臂、flush 日志
    print(f"[app pid={os.getpid()}] clean exit", flush=True)
    sys.exit(0)


if os.environ.get("HANDLE_SIGTERM") == "1":
    signal.signal(signal.SIGTERM, on_sigterm)

print(f"[app pid={os.getpid()} ppid={os.getppid()}] started, HANDLE_SIGTERM={os.environ.get('HANDLE_SIGTERM')}", flush=True)
while True:
    time.sleep(1)
