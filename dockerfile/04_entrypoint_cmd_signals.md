# 第 4 章：ENTRYPOINT、CMD、PID 1 与信号

这一章的内容平时很少有人在意，直到某天 `docker stop` 一个服务要卡 10 秒、
或者服务被强杀导致数据库文件损坏、机械臂没有正常停下……

## 4.1 ENTRYPOINT 与 CMD 怎么组合

容器启动时执行的命令 = **ENTRYPOINT + CMD**（拼接成一个 argv）。

| Dockerfile | `docker run img` | `docker run img foo bar` |
|---|---|---|
| `CMD ["gunicorn", ":8000"]` | `gunicorn :8000` | `foo bar`（CMD 被整个替换） |
| `ENTRYPOINT ["run.sh"]` | `run.sh` | `run.sh foo bar` |
| `ENTRYPOINT ["run.sh"]` + `CMD ["--help"]` | `run.sh --help` | `run.sh foo bar` |
| `docker run --entrypoint bash img` | `bash`（**同时清空镜像里的 CMD**） | `bash foo bar` |

对照仓库：

| 镜像 | 设置 | 使用方式 |
|---|---|---|
| robot-data-platform | 只有 `CMD [gunicorn ...]` | `docker run rdp` 跑服务；`docker run rdp python manage.py migrate` 做迁移 |
| brain-service | 只有 `ENTRYPOINT [run_brain.sh]` | `docker run brain-service launch-all` → `run_brain.sh launch-all` → `launch launch-all` |
| infra-bin / model-bin | `ENTRYPOINT ["/bin/bash", "entrypoint.sh"]` | `docker run infra-bin bin/some_tool --flag` → 执行 bazel 产物 |
| universal-uploader | 只有 `CMD [launcher]` | compose 里用 `command:` 覆盖成一段 bash 脚本 |

**设计原则：**
- 镜像就是"一个程序"（如 brain-service）→ 用 `ENTRYPOINT`，参数从 `docker run` 后面传。
- 镜像是"一个环境，默认跑某个程序"（如 rdp，可能还要跑 migrate/shell）→ 用 `CMD`。
- "包装脚本 + 任意命令"（如 infra-bin）→ `ENTRYPOINT` 是包装脚本，脚本最后 `exec "$@"`。

## 4.2 exec 形式 vs shell 形式

```dockerfile
CMD ["gunicorn", "--bind", ":8000"]     # exec 形式（JSON 数组）：直接执行 gunicorn
CMD gunicorn --bind :8000               # shell 形式：执行 /bin/sh -c "gunicorn --bind :8000"
```

shell 形式的两个问题：
1. **PID 1 是 `/bin/sh`**，不是你的程序，信号转发出问题（见 4.3）。
   实测 `python:3.11-slim` 里 `CMD python app.py`：PID 1 是 `/bin/sh -c python app.py`（dash），python 的 PID 是 7 ——
   dash 并没有把唯一的命令 exec 掉。`docker build --check` 会对此报 `JSONArgsRecommended`。
2. **shell 形式的 ENTRYPOINT 会忽略 CMD 和 `docker run` 后面的所有参数。**

仓库里所有 ENTRYPOINT/CMD 都用了 exec 形式 ✅。

注意 JSON 数组必须用**双引号**，`CMD ['gunicorn']` 不是合法 JSON，
Docker 会退回把它当 shell 形式处理（较新版本会给警告）。

## 4.3 PID 1 的特殊性：为什么 `docker stop` 会卡 10 秒

`docker stop` 的流程：

```
发送 SIGTERM 给容器的 PID 1
   │
   ├── PID 1 在 10 秒内退出 → 正常停止
   └── 10 秒后仍未退出 → 发送 SIGKILL（强杀，不可捕获）
```

（`docker stop -t 30` 可以改等待时长，compose 里是 `stop_grace_period`。）

两个 Linux 机制决定了结果：

**① 内核对 PID 1 的特殊保护**：在 PID namespace 里，PID 1 只会收到**它自己注册了处理函数的**信号。
没有 handler 的 SIGTERM 会被**直接忽略**，而不是执行默认的"终止进程"动作（SIGKILL 例外）。
（`man 7 pid_namespaces`）

**② bash 不会把信号转发给子进程。**

现在看两个仓库里的 ENTRYPOINT 脚本。

### 正例：`infra-bin/entrypoint.sh`

```bash
#!/bin/bash
export RUNFILES_DIR=$1.runfiles
exec "$@"
```

- `exec`：用目标程序**替换**当前 bash 进程，目标程序直接成为 PID 1，SIGTERM 直接到它手里。
- `"$@"`（带引号）：原样传递所有参数，含空格的参数也不会被拆开。
- `RUNFILES_DIR=$1.runfiles`：bazel 产物 `foo` 的依赖都在 `foo.runfiles/` 目录里，
  第一个参数就是要执行的二进制，所以拼出它的 runfiles 路径。这是一个"在 exec 前做点准备工作"的标准包装脚本。

### 反例：`brain-service/scripts/run_brain.sh`

```bash
#!/bin/bash
python /opt/generate_ueye_conf.py
service ueyeethdrc start
python /root/brain-service/bazel-bin/covariant/service/launch "$@"      # ← 没有 exec
```

进程树：

```
PID 1  bash run_brain.sh            ← 收到 SIGTERM；没有 trap → 被内核忽略
  └── PID N  python launch ...      ← 永远收不到 SIGTERM
        └── 各个子服务...
```

`docker stop brain-service` 的实际效果：**等满 10 秒，然后所有进程被 SIGKILL**。
launch 里任何"收到 SIGTERM 时优雅关闭"的代码（停相机、断开机械臂、flush 日志、关数据库连接）都不会执行。

修复只需要一个词：

```bash
exec python /root/brain-service/bazel-bin/covariant/service/launch "$@"
```

但还有第二个坑：**exec 之后 python 成为 PID 1，而 Python 默认没有给 SIGTERM 注册 handler**
（只给 SIGINT 注册了，用来抛 KeyboardInterrupt）。根据机制 ①，SIGTERM 还是会被忽略。
所以要么在程序里处理 SIGTERM：

```python
import signal, sys
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # 或者触发你的清理逻辑
```

要么让一个**专门的 init 进程**当 PID 1：

```bash
docker run --init ...          # Docker 自带 tini 作为 PID 1（容器里显示为 /sbin/docker-init）
```

```yaml
services:
  brain:
    init: true                 # compose 等价写法
```

tini 做两件事：把收到的信号转发给子进程；回收僵尸进程（PID 1 的另一个职责：
孤儿进程会被挂到 PID 1 下，如果 PID 1 不 `wait()` 它们，就会积累僵尸进程）。
子进程不是 PID 1，SIGTERM 的默认动作（终止）就恢复生效了。

**结论：容器的 ENTRYPOINT 脚本最后一行用 `exec`；能加 `--init` 就加。**

lab04 在 Jetson Thor 上的实测（5 种组合，`docker stop` 耗时）：

| 组合 | PID 1 | 耗时 | ExitCode | 程序执行了清理吗 |
|---|---|---|---|---|
| 没有 exec（= run_brain.sh） | bash | 10.9s | 137 | ❌ |
| exec，程序没有 handler | python | 10.8s | 137 | ❌ |
| **exec + 程序处理 SIGTERM** | python | **1.2s** | **0** | ✅ |
| 没有 exec + `--init` | docker-init | 0.8s | 143 | ❌（bash 先死，python 被连带 SIGKILL） |
| exec + `--init`，没有 handler | docker-init | 0.8s | 143 | ❌（默认动作直接终止） |

注意第 4 行：**`--init` 不能代替 `exec`** —— 停得快，但程序的清理代码根本没机会执行。

### STOPSIGNAL 会继承（lab06 中踩到的坑）

`docker stop` 发送的不一定是 SIGTERM，而是镜像的 `STOPSIGNAL`，**它会从基础镜像继承**。
例如官方 `postgres` 镜像设置了 `STOPSIGNAL SIGINT`。lab06 借用 postgres 镜像跑一个 bash 循环，
加了 `init: true` 仍然要 10.9 秒才能停下：非交互式 bash 会捕获 SIGINT，并且等前台子进程也因 SIGINT 退出时才跟着退出；
而 tini 只把信号发给 bash、不发给正在运行的 `sleep`，于是循环照常继续。显式写 `stop_signal: SIGTERM` 后降到 0.8 秒。

```bash
docker image inspect -f '{{.Config.StopSignal}}' <镜像>      # 空 = 默认 SIGTERM
```

### 同样的问题出现在 compose 文件里

`universal-uploader/assert/docker-compose.yaml`：

```yaml
command: |
    bash -c "
        export UNIVERSAL_UPLOADER_POSTGRES_USER=$$(cat /run/secrets/universal_uploader_postgres_user) &&
        export UNIVERSAL_UPLOADER_POSTGRES_PASSWORD=$$(cat /run/secrets/universal_uploader_postgres_password) &&
        /universal_uploader/launcher --config /var/covariant/universal_uploader/config.yaml
    "
```

bash 是 PID 1，launcher 是它的子进程。建议最后一行改成 `exec /universal_uploader/launcher ...`
（不要依赖 bash 自己的"最后一条命令自动 exec"优化，它只在部分简单情况下触发）。

```yaml
entrypoint: ["/bin/bash"]
command: ["-c", "while true; do finalize_orphaned_files; sleep 120; done"]
```

这个是 bash 循环本身作为 PID 1，`docker stop` 一样会等 10 秒；这里加 `init: true` 就够了。

### 写对了的：gunicorn

rdp 的 `CMD ["gunicorn", ...]`：exec 形式，gunicorn 直接是 PID 1，而且 gunicorn 自己处理 SIGTERM（优雅关闭 worker）。✅

## 4.4 案例精读：`dev.sh` 如何覆盖 ENTRYPOINT

```bash
docker run ... \
    --entrypoint /root/brain-service/scripts/docker_entrypoint.py \
    -v "$REPO_DIR:/root/brain-service" \
    ... \
    "$IMAGE_NAME" \
    --command bash -c "$CMD"
```

拆解：

1. `--entrypoint` 把 dev 镜像原本的 ENTRYPOINT 换成 `docker_entrypoint.py`。
2. 这个脚本路径在 `/root/brain-service/` 下，而这个目录是**从宿主机挂进来的源码**（`-v $REPO_DIR:/root/brain-service`）。
   也就是说 **ENTRYPOINT 脚本来自你本地的源码**，改了立即生效，不用重建镜像。巧妙。
3. 镜像名后面的 `--command bash -c "$CMD"` 成为 CMD，拼到 ENTRYPOINT 后面。

`docker_entrypoint.py` 做的事情（容器"开机初始化"）：往 `.bashrc` 写 PATH/PYTHONPATH、alias；
可选启动 ueye 相机守护进程；安装 `bb` 工具；非生产环境建测试数据库；改 sshd 端口并启动 ssh；
拷贝 third_party；最后执行 `--command` 指定的命令。

### ⚠ 一个参数解析的坑（已实测）

```python
parser.add_argument("-c", "--command", type=str, default="/bin/bash")
```

`-c` 和 `--command` **是同一个选项**。所以 `--command bash -c "$CMD"` 在 argparse 看来是
"先把 command 设为 `bash`，再把 command 覆盖为 `$CMD`"。用 Python argparse 实测：

```
['--command', 'bash', '-c', 'bash']                                    -> 'bash'
['--command', 'bash', '-c', 'bb covariant/service/launch launch-all']  -> 'bb covariant/service/launch launch-all'
```

然后 `subprocess.check_call(shlex.split(args.command))` 直接执行，**中间根本没有 bash**。
作者的本意显然是 `bash -c "$CMD"`，现在能工作只是因为 `$CMD` 通常是一个简单命令。
如果你执行 `dev.sh -- "cd covariant && bb test"`，`shlex.split` 会把 `cd`、`&&` 当成普通参数，执行失败。

修法：`--command "bash -c '$CMD'"`，或者让 `docker_entrypoint.py` 用 `["bash", "-c", args.command]` 执行。

### 另一个小问题：`.bashrc` 只追加不去重

`docker_entrypoint.py` 每次启动都往 `/root/.bashrc` 追加内容。`dev.sh` 用了 `--rm`，
每次都是新容器所以没问题；但如果有人去掉 `--rm`，然后反复 `docker start` 同一个容器，`.bashrc` 会越来越长。
容器初始化脚本应该是**幂等**的。

## 4.5 用 `docker exec` 进入运行中的容器

```bash
docker exec -it <容器> bash
```

`exec` 在已运行的容器里**另起一个进程**，不是 PID 1，不经过 ENTRYPOINT。
`dev.sh` 的"容器已存在"分支用的就是它：

```bash
docker exec -it "$CONTAINER_NAME" "$CMD"
```

注意这里把 `$CMD` 整个作为**一个参数**（可执行文件名）传给 docker exec。
`CMD=bash` 时没问题；`CMD="bb covariant/service/launch launch-all"` 时 docker 会去找一个
名为 `bb covariant/service/launch launch-all` 的可执行文件，报 `executable file not found`。
应写成 `docker exec -it "$CONTAINER_NAME" bash -c "$CMD"`。

## 4.6 动手

做 [code/lab04_signals](code/lab04_signals/)：四种 ENTRYPOINT 写法，用 `time docker stop` 测停止耗时，
直观看到"没有 exec"和"PID 1 没有 handler"的后果。

下一章：[05_docker_run_in_depth.md](05_docker_run_in_depth.md)
