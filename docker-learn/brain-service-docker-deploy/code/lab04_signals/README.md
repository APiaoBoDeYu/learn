# lab04：ENTRYPOINT、PID 1 与信号

对应第 4 章。

## 文件

| 文件 | 模仿 |
|---|---|
| `app.py` | 一个长期运行的服务；`HANDLE_SIGTERM=1` 时注册 SIGTERM 处理函数，做"清理"后退出 |
| `entrypoint_noexec.sh` | `brain-service/scripts/run_brain.sh`：准备工作 + 直接调用程序 |
| `entrypoint_exec.sh` | `infra-bin/entrypoint.sh`：准备工作 + `exec` |

## 运行

```bash
cd lab04_signals
./run.sh
```

脚本对 5 种组合分别 `docker run -d` → 等 2 秒 → 计时 `docker stop`，并打印 PID 1 的命令行、退出码和容器日志。

## 实测结果（Jetson Thor, Docker 29.1.4）

与事先根据 Linux/Docker 机制推导的预期**完全一致**：

| # | 组合 | PID 1 | stop 耗时 | ExitCode | 日志里有 "clean exit" 吗 |
|---|---|---|---|---|---|
| 1 | 没有 exec（= run_brain.sh） | `/bin/bash /app/entrypoint_noexec.sh` | **10.9s** | 137 | ❌ |
| 2 | exec，程序没有 handler | `python /app/app.py` | **10.8s** | 137 | ❌ |
| 3 | exec + 程序处理 SIGTERM | `python /app/app.py` | 1.2s | 0 | ✅ |
| 4 | 没有 exec + `--init` | `/sbin/docker-init -- /app/entrypoint_noexec.sh` | 0.8s | 143 | ❌ |
| 5 | exec + `--init`，没有 handler | `/sbin/docker-init -- /app/entrypoint_exec.sh` | 0.8s | 143 | ❌（被默认动作直接终止） |

（`docker stop` 本身有约 0.8s 的固定开销；情况 3 多出的 0.4s 是 app.py 里模拟清理的 `sleep(0.5)`。）

各情况的容器日志：

```
1) [wrapper pid=1] preparing...
   [app pid=7 ppid=1] started, HANDLE_SIGTERM=1          ← 注册了 handler，但信号根本没到它这里
2) [wrapper pid=1] preparing...
   [app pid=1 ppid=0] started, HANDLE_SIGTERM=0          ← exec 后 python 就是 PID 1
3) [app pid=1 ppid=0] started, HANDLE_SIGTERM=1
   [app pid=1] got SIGTERM, cleaning up...
   [app pid=1] clean exit
4) [wrapper pid=7] preparing...
   [app pid=8 ppid=7] started, HANDLE_SIGTERM=1          ← 停得快，但没有 cleaning up
5) [wrapper pid=7] preparing...
   [app pid=7 ppid=1] started, HANDLE_SIGTERM=0
```

解读：

- **1 和 2 都是 10 秒 + 137**（137 = 128 + 9，被 SIGKILL）。PID 1 没有为 SIGTERM 注册 handler，
  内核直接忽略这个信号；Docker 等满 10 秒后发 SIGKILL。
  情况 1 里 python 虽然注册了 handler，但 SIGTERM 只发给了 PID 1（bash），bash 不转发。
- **3 是唯一真正"优雅关闭"的组合**：信号直达程序，程序执行了清理逻辑并以 0 退出。
- **4 很有迷惑性**：停得很快，但看日志会发现 python 的清理代码**没有执行**。
  tini 只把 SIGTERM 转发给它的直接子进程 bash，bash（不再是 PID 1）按默认动作立即死亡，
  tini 随之退出，容器内剩下的 python 在 PID namespace 销毁时被内核 SIGKILL。
  也就是说 **`--init` 不能代替 `exec`**。
- **5**：tini 把 SIGTERM 转给 python；python 不是 PID 1，SIGTERM 的默认动作（终止）生效，所以快速退出，
  但没有执行任何清理。143 = 128 + 15（被 SIGTERM 终止）。

**结论：包装脚本用 `exec`，程序处理 SIGTERM（3）；如果没法改程序，至少 `exec + --init`（5）让它能被及时停掉。**

### 补充：不在容器里也能观察到的部分

"bash 不转发信号"和"exec 让程序直接收信号"不依赖 namespace，可以直接测：

```
# entrypoint_exec.sh，向包装脚本的 PID 发 SIGTERM
[wrapper pid=101472] preparing...
[app pid=101472 ppid=101467] started, HANDLE_SIGTERM=1      ← 同一个 PID：exec 替换了 bash
[app pid=101472] got SIGTERM, cleaning up...
[app pid=101472] clean exit

# entrypoint_noexec.sh，向包装脚本的 bash 发 SIGTERM
wrapper bash=101681 python=101683
after SIGTERM->wrapper: wrapper alive=no, python alive=yes  ← bash 死了，python 没收到信号，成了孤儿
```

这正是情况 4 的机制：tini 把信号交给 bash，bash 自己死了，python 什么都不知道。

## 延伸

```bash
# 看容器内进程树
docker run -d --name lab04-tree -e HANDLE_SIGTERM=1 lab04
# python-slim 里没有 ps（实测报 executable file not found），所以有后面的备用写法
docker exec lab04-tree ps -ef 2>/dev/null || docker exec lab04-tree sh -c 'for p in /proc/[0-9]*; do echo "$(basename $p) $(tr "\0" " " < $p/cmdline)"; done'
docker rm -f lab04-tree

# 改等待时间
docker run -d --name lab04-t lab04 && time docker stop -t 3 lab04-t && docker rm lab04-t

# 换个停止信号：python 对 SIGINT 默认有 handler（KeyboardInterrupt），所以即便是 PID 1 也会响应
docker run -d --name lab04-int --stop-signal SIGINT --entrypoint /app/entrypoint_exec.sh lab04
time docker stop lab04-int; docker logs lab04-int; docker rm lab04-int
```

最后一个实验的实测：0.8s 停止，ExitCode=130（128+2），日志末尾是 `KeyboardInterrupt` 的 traceback。
这也是 Dockerfile `STOPSIGNAL` 指令的用途。
