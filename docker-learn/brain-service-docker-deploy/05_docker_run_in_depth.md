# 第 5 章：`docker run` 参数全解 —— 精读 `dev.sh`

你已经会 `docker run -it image bash`。这一章把 `brain-service/scripts/dev.sh` 里那条长命令逐项拆开，
每个参数讲清楚**它解除了哪种隔离、为什么这里需要、代价是什么**。

```bash
docker run --network host --ipc host --privileged \
    --name "$CONTAINER_NAME" \
    --entrypoint /root/brain-service/scripts/docker_entrypoint.py \
    -v /dev:/dev \
    --gpus all \
    -v "$LOCAL_DATA_DIR:/var/covariant" \
    -v "$REPO_DIR:/root/brain-service" \
    -v "$HOME/.cache:/root/.cache" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    "${DOCKER_EXTRA_PARAMS[@]}" \
    "${ENV_ARGS[@]}" \
    --rm \
    -it \
    "$IMAGE_NAME" \
    --command bash -c "$CMD"
```

CI（`bamboo-specs/bamboo.yaml` 的 "X86 Docker building"）跑构建任务时用的几乎是同一组参数：

```yaml
docker:
  image: robot-harbor.geekplus.com/gino/brain-service-dev:v1.1.0
  volumes:
    "/dev": "/dev"
    "/var/run/docker.sock": "/var/run/docker.sock"
    "/var/covariant/": "/var/covariant/"
  docker-run-arguments: ["--net=host", "--gpus=all", "--ipc=host", "--privileged"]
```

## 5.1 隔离的维度

回顾第 0 章：容器 = 被 namespace 和 cgroup 圈起来的进程。`docker run` 的很多参数就是在逐项"拆墙"：

| 参数 | 拆掉的隔离 | 一句话 |
|---|---|---|
| `--network host` | network namespace | 容器直接用宿主机网卡和端口 |
| `--ipc host` | IPC namespace | 共享宿主机的共享内存 `/dev/shm` |
| `--pid host` | PID namespace | 能看到/操作宿主机所有进程（compose 里 finalize 服务用） |
| `--privileged` | capabilities、设备 cgroup、seccomp、AppArmor | 几乎等于宿主机 root |
| `-v /dev:/dev` | mount namespace（部分） | 看到宿主机所有设备节点 |
| `--gpus all` | 设备 cgroup（GPU 部分） | 通过 nvidia-container-toolkit 注入 GPU 设备和驱动库 |

## 5.2 网络：`--network host`

**默认（bridge 网络）**：容器有自己的 IP（如 172.17.0.2），宿主机做 NAT。
要从外部访问容器里的服务，必须 `-p 宿主机端口:容器端口` 映射。

**host 网络**：容器和宿主机共享同一个网络栈。容器里监听 8000，宿主机的 8000 就被占用了。
`-p` 参数无效（也不需要）。

dev.sh / CI / compose 里大量使用 host 网络的原因：
- 机器人工作站上，服务要直接和**局域网里的硬件**通信（ABB 机械臂控制器 `ABB_VC_IP`、
  GigE 相机、Amcrest 网络摄像头 `192.168.11.250`）。bridge 网络下有 NAT，广播/组播发现协议（比如 GigE Vision 相机发现）通常无法工作。
- 很多服务之间用 `localhost:端口` 互相访问，host 网络下最简单。
- 性能：没有 NAT 和 veth 开销。

代价：**端口冲突**。这就是 dev.sh 有 `--port` 参数的原因：

```bash
SSH_PORT="222"
--port) SSH_PORT="$2" ...
```

`docker_entrypoint.py` 会把 sshd 端口改成 `$SSH_PORT`。同一台机器上开两个 dev 容器，
如果都用 222，第二个 sshd 起不来 —— host 网络下它们抢的是同一个宿主机端口。

（注：Docker Desktop for Mac/Windows 上 host 网络的语义不同，这里只讨论 Linux。）

## 5.3 共享内存：`--ipc host`

Docker 默认给每个容器一个只有 **64MB** 的 `/dev/shm`（Thor 上实测 `df -h /dev/shm`：`64M`）。PyTorch DataLoader 多 worker、
多进程间共享图像帧、ROS2 共享内存传输等，都会用 `/dev/shm`，64MB 很快不够，报
`Bus error` 或 `unable to write to file </torch_xxx>`。

两种解决办法：

```bash
docker run --ipc host ...          # 直接用宿主机的 /dev/shm（通常是内存的一半）
docker run --shm-size 8g ...       # 保持隔离，只把容量调大
```

`--shm-size` 更"干净"；`--ipc host` 还能让容器和宿主机上的进程通过共享内存通信。

## 5.4 硬件访问：`--privileged` 与 `-v /dev:/dev`

默认容器：
- 只能看到少数几个设备（`/dev/null`、`/dev/zero`、`/dev/tty` 等）；
- root 只有一部分 capabilities（不能 mount、不能改网络配置、不能加载内核模块……）；
- 有 seccomp 系统调用过滤。

`--privileged` 把这些**全部关掉**：所有设备可访问、所有 capabilities、无 seccomp/AppArmor。
对于要访问 USB 相机、串口、GPIO、CAN 总线等各种硬件的机器人开发容器来说，这是最省事的做法。

`-v /dev:/dev` 的额外作用：`--privileged` 只在**容器启动那一刻**把设备节点建出来，
之后热插拔的 USB 设备不会出现在容器里。把宿主机的 `/dev` 整个挂进来，热插拔的设备就能实时可见。

**代价：`--privileged` 的容器和宿主机 root 之间几乎没有安全边界。** 开发环境可以接受，
生产环境建议改成精确授权：

```bash
docker run --device /dev/ttyUSB0 --device /dev/video0 \
           --cap-add SYS_NICE --cap-add NET_ADMIN ...
```

## 5.5 GPU：`--gpus all`

`--gpus` 依赖宿主机安装了 NVIDIA 驱动和 **nvidia-container-toolkit**。它在容器启动时注入：
GPU 设备节点（`/dev/nvidia*`）、宿主机的驱动用户态库（`libcuda.so` 等）、`nvidia-smi`。

所以：**镜像里只需要 CUDA 运行库（toolkit），不需要也不应该装 GPU 驱动**。
驱动版本由宿主机决定，这也是"镜像的 CUDA 版本不能高于宿主机驱动支持的版本"的原因。

```bash
docker run --gpus all ...                 # 所有 GPU
docker run --gpus '"device=0,1"' ...      # 指定 GPU
docker run --runtime nvidia ...           # 老写法，使用 nvidia runtime
```

**Jetson 上的特殊情况**：Jetson 是集成 GPU（和 CPU 共享内存），历史上 L4T 的标准写法是 `--runtime nvidia`，
JetPack 会把驱动库通过 CSV 清单挂进容器。在 Jetson Thor（L4T R38.4.0）上实测：

- `/etc/docker/daemon.json` 里配置了 `"default-runtime": "nvidia"`；
- `docker run --gpus all debian:bookworm-slim`：容器里能看到 6 个 `/dev/nvidia*` 设备节点，`libcuda` 也被注入了；
- `docker run --runtime runc debian:bookworm-slim`（绕过 nvidia runtime）：0 个 nvidia 设备。

所以在 Thor 上 `--gpus all` 是可用的。较老的 JetPack 上如果 `--gpus all` 报错，换成 `--runtime nvidia`。

## 5.6 挂载：bind mount / volume / tmpfs

### 三种挂载

| 类型 | 写法 | 数据在哪 | 典型用途 |
|---|---|---|---|
| **bind mount** | `-v /宿主机/绝对路径:/容器路径` | 宿主机上你指定的目录 | 源码、配置、数据目录、设备 |
| **named volume** | `-v 卷名:/容器路径` | Docker 管理（`/var/lib/docker/volumes/`） | 数据库数据（Docker 管理生命周期） |
| **tmpfs** | `--tmpfs /容器路径` | 内存 | 临时文件 |

`-v` 冒号左边是**绝对路径**就是 bind mount，是**名字**就是 named volume。
更清晰的长写法是 `--mount type=bind,source=...,target=...`（compose 里 db 服务就用了长写法）。
`-v` 和 `--mount` 有个关键区别：bind mount 的源路径不存在时，`-v` 会**自动创建一个 root 拥有的空目录**，
`--mount` 会直接报错。`deploy.sh` 要先把十几个目录 `mkdir -p` 好，就是为了避免出现一堆 root 所属的空目录。

### dev.sh 的四个 bind mount

| 挂载 | 目的 |
|---|---|
| `$LOCAL_DATA_DIR:/var/covariant` | 运行数据（日志、事件、配置、视频）落在宿主机上，容器删了不丢。整个服务体系约定数据都在 `/var/covariant` 下，compose 文件里也是 |
| `$REPO_DIR:/root/brain-service` | **用宿主机源码覆盖镜像里的源码**（见下） |
| `$HOME/.cache:/root/.cache` | **bazel / pip 缓存持久化**。bazel 的 output base 在 `/root/.cache/bazel`，挂出来后容器 `--rm` 删掉，下次进来增量编译还能用 |
| `/var/run/docker.sock` | 容器里能操作宿主机的 docker（见 5.7） |

**挂载会遮盖镜像里的同名目录。** dev 镜像里 `/root/brain-service` 可能本来就有内容，
`-v $REPO_DIR:/root/brain-service` 之后，容器里看到的是宿主机目录的内容，镜像里那份被"盖住"了（不是删除，卸载后还在）。
开发容器靠这一点实现"改宿主机代码、容器里立即生效"。

### 文件属主问题

容器里默认是 root。写进 bind mount 的文件，在宿主机上也是 root 所属（UID 0）。
宿主机普通用户之后就删不掉/改不了这些文件。`deploy.sh` 里大量 `sudo -S mkdir` 的逻辑，
就是在处理 `LOCAL_DATA_DIR` 属主不是当前用户的情况。

解决方式：

```bash
docker run --user "$(id -u):$(id -g)" ...     # 以宿主机当前用户身份运行
```

但很多开发镜像假设自己是 root（比如往 `/root/.bashrc` 写东西、`service ssh start`），
用 `--user` 会失败。这也是机器人团队开发容器普遍以 root 运行的原因。

## 5.7 挂载 docker.sock：Docker-out-of-Docker

```bash
-v /var/run/docker.sock:/var/run/docker.sock
```

容器里的 `docker` 命令连的是**宿主机的 daemon**。在容器里 `docker build`、`docker run` 创建的是宿主机上的"兄弟容器"，
而不是嵌套在当前容器里的"子容器"。CI 就是这样在 dev 容器里执行 `build_docker.sh` 来构建镜像的。

**最大的坑：路径是按宿主机解释的。** 在容器里执行：

```bash
docker run -v /root/brain-service:/src some-image
```

daemon 在宿主机上运行，它去找的是**宿主机的** `/root/brain-service`，不是你当前容器里的那个目录。
如果容器内外路径不一致，挂进去的就是错误的目录（或空目录）。

这正是 bamboo.yaml 里这行挂载看起来很奇怪的原因：

```yaml
"/data/bamboo-agent/bamboo-agent-home/xml-data/build-dir/_git-repositories-cache/": "/data/bamboo-agent/.../_git-repositories-cache/"
```

**容器内外用同一个路径**，这样无论在哪里引用这个路径，含义都一致。

`docker build` 不受这个影响：CLI 会把上下文打包发给 daemon，上下文路径在容器里解释就对了。

安全提示：**能访问 docker.sock 就等于拥有宿主机 root**（随便 `docker run -v /:/host --privileged` 就能读写宿主机所有文件）。

## 5.8 环境变量

```bash
docker run -e KEY=value ...       # 显式指定值
docker run -e KEY ...             # 不写值：从当前 shell 继承（未设置则不传）
docker run --env-file .env ...    # 从文件读取
```

dev.sh 的做法：

```bash
ENV_VARS=("ABB_VC_IP" "DISPLAY" ... "AWS_SECRET_ACCESS_KEY" ... "SSH_PORT")
ENV_ARGS=()
for VAR in "${ENV_VARS[@]}"; do
    if [[ -n "${!VAR}" ]]; then                 # ${!VAR}：bash 间接引用，取"名为 $VAR 的变量"的值
        ENV_ARGS+=(--env "$VAR=${!VAR}")
    fi
done
```

效果等价于对每个变量写 `-e VAR`（未设置就不传）。白名单的好处是不会把宿主机所有环境变量都塞进容器。
这些变量大多来自 `docker-deploy/env.sh`，用法是先 `source env.sh` 再执行 `dev.sh`。

注意：
- **通过 `-e` 传的值能被任何有 docker 权限的人用 `docker inspect` 看到**（包括 `AWS_SECRET_ACCESS_KEY`）。
  compose 的 universal-uploader 对数据库密码用了 secrets 文件（第 6 章），更安全。
- 传了 `DISPLAY` 想在容器里开 GUI 窗口，一般还需要 `-v /tmp/.X11-unix:/tmp/.X11-unix` 和宿主机执行 `xhost +local:`；
  如果 DISPLAY 是 ssh X 转发的 `localhost:10.0`，因为用了 host 网络，可以直接连上。

## 5.9 生命周期参数

| 参数 | 含义 |
|---|---|
| `-it` | `-i` 保持 stdin 打开 + `-t` 分配终端。交互式 shell 必须有 |
| `-d` | 后台运行（detach） |
| `--rm` | 容器退出后自动删除（包括它的可写层）。开发容器常用 |
| `--name` | 指定名字，之后可以 `docker exec/logs/stop <名字>` |
| `--restart always / unless-stopped / on-failure` | 进程退出或宿主机重启后自动拉起。生产服务必备（compose 里用 `restart:`） |
| `--memory 8g --cpus 4` | cgroup 资源限制 |

### dev.sh 的"复用已有容器"逻辑

```bash
if [[ $(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}") == "$CONTAINER_NAME" ]]; then
  docker exec -it "$CONTAINER_NAME" "$CMD"
else
  docker run ... --name "$CONTAINER_NAME" ... --rm -it ...
fi
```

思路：第一个终端 `dev.sh` 启动容器，再开一个终端 `dev.sh`，就 `exec` 进同一个容器。几个细节：

- `--filter name=xxx` 是**子串/正则匹配**。如果同时存在 `brain-dev-root` 和 `brain-dev-root2`，
  输出是两行，`==` 比较失败，脚本会尝试再 `docker run` 一个同名容器，报 `Conflict. The container name ... is already in use`。
  精确匹配要写 `--filter "name=^${CONTAINER_NAME}$"`。
- 因为有 `--rm`，**第一个终端退出 shell，容器就被删了**，第二个终端里的会话也随之断开。
- `docker exec -it name "$CMD"` 的引号问题见第 4 章 4.5。

## 5.10 日常排错命令

```bash
docker ps -a                              # 包括已退出的容器
docker logs -f --tail 100 <容器>           # 看 PID 1 的 stdout/stderr
docker inspect <容器>                      # 全部配置：挂载、网络、环境变量、ExitCode、OOMKilled
docker inspect -f '{{.State.ExitCode}} {{.State.OOMKilled}}' <容器>
docker top <容器>                          # 容器里的进程（宿主机视角的 PID）
docker stats                              # 实时 CPU/内存
docker cp <容器>:/path ./                  # 拷文件出来
docker diff <容器>                         # 可写层相对镜像改了哪些文件
docker run --rm -it --entrypoint bash <镜像>   # 镜像的 ENTRYPOINT 起不来时，绕过它进去看
```

最后一条特别常用：服务容器启动即退出、`docker logs` 又看不出原因时，用 `--entrypoint bash` 进去手动执行一遍。

## 5.11 一个练习

不看答案，说出下面每个参数在你的场景里是否需要，以及替代方案：

```bash
docker run --network host --ipc host --privileged -v /dev:/dev --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock ...
```

假设场景：一个纯推理服务，只用 GPU，对外暴露 HTTP 8080 端口，不访问其他硬件。

<details>
<summary>参考答案</summary>

- `--network host` → 不需要，用 `-p 8080:8080`
- `--ipc host` → 一般不需要；如果用多进程预处理，改用 `--shm-size 2g`
- `--privileged`、`-v /dev:/dev` → 不需要
- `--gpus all` → 需要（或 `--gpus '"device=0"'`）
- docker.sock → 不需要，且挂载有安全风险
- 另外建议加 `--restart unless-stopped`、`--init`

</details>

下一章：[06_compose.md](06_compose.md)
