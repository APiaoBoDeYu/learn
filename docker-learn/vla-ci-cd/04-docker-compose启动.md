# 04 docker-compose.yml：在机器人上启动容器

文件：`embodied_algo/body/gino/docker-compose/docker-compose.yml`

## docker-compose 是什么

先打个比方：

- **镜像**像一份预制菜。代码、Python、依赖库都已经打包好，在哪台机器上加热（运行）都是同一个味道。
- **容器**是正在加热、正在吃的那一份。
- **docker-compose.yml** 是一张套餐菜单。上面写着有哪几道菜、每道菜怎么做、谁先上。

这个项目有两道菜：

- `vla-body`：机器人的"身体"，负责电机驱动和硬件。
- `vla-brain`：机器人的"大脑"，在 8000 端口提供模型推理服务。

不用 compose 的话，要手敲两条很长的 `docker run`，每条带几十个参数，还要记得先启动 body 再启动 brain。compose 把这些参数写进一个 YAML 文件，用一条命令统一管理：

```bash
docker compose up -d        # 拉取镜像（本地没有时）并在后台启动所有服务
docker compose pull         # 拉取最新镜像（tag 被覆盖过时需要）
docker compose ps           # 查看状态
docker compose logs -f vla-brain
docker compose exec vla-body bash
docker compose down         # 停止并删除容器
```

文件里的每个 `services` 条目对应一个容器。

compose 带来的好处：

1. 一条命令管多个容器，不用记参数。
2. 配置即文档。部署方式写在文件里，可以进 git、做 review，每台机器上的部署都一样。
3. 代码和数据分开。代码在镜像里，换版本只改 `VLA_VERSION`；模型、日志放在宿主机目录，不受影响。

## 启动方式

```bash
cd body/gino/docker-compose
VLA_VERSION=2.1.0rc docker compose up -d
```

也可以把变量写进同目录下的 `.env` 文件，compose 会自动读取：

```
VLA_VERSION=2.1.0rc
VLA_DATA_ROOT=/data/vla
```

## 变量替换

```yaml
image: robot-harbor.geekplus.com/gino/vla-body:${VLA_VERSION:?set VLA_VERSION, e.g. 2.1.0rc}
source: ${VLA_DATA_ROOT:-/data/vla}/checkpoints
```

| 写法 | 含义 |
|---|---|
| `${VAR:?消息}` | VAR 未设置或为空时，报错并显示消息，compose 不启动 |
| `${VAR:-默认值}` | VAR 未设置或为空时，使用默认值 |

`VLA_VERSION` 必须填，并且要和 CI 推送的 tag 一致。**这是 compose 和 CI 之间唯一的联系。**

## 配置项总表

以 `vla-brain` 为例，`vla-body` 大部分相同。后面几节对其中容易看不懂的项展开讲。

| 配置 | 含义 | 等价的 docker run 参数 |
|---|---|---|
| `image` | 用哪个镜像 | 镜像名 |
| `container_name` | 容器名 | `--name` |
| `command` | 覆盖镜像的 CMD | 镜像名后面的参数 |
| `user: "0:0"` | 以 root 运行 | `--user 0:0` |
| `runtime: nvidia` | 使用 NVIDIA 容器运行时，容器里能用 GPU | `--runtime nvidia` |
| `network_mode: host` | 和宿主机共用网络，不做端口映射 | `--net=host` |
| `ipc: host` | 和宿主机共用共享内存，PyTorch 多进程和共享内存相机需要 | `--ipc=host` |
| `privileged: true` | 获得几乎所有宿主机权限，用于访问硬件 | `--privileged` |
| `stdin_open` + `tty` | 保持标准输入打开并分配终端 | `-it` |
| `restart: unless-stopped` | 容器退出或机器重启后自动拉起，手动 stop 的除外 | `--restart unless-stopped` |
| `ulimits.nofile` | 最大打开文件数调到 64000 | `--ulimit nofile=64000:64000` |
| `logging` | 容器日志单文件最大 20 MB，保留 5 个 | `--log-driver` / `--log-opt` |
| `depends_on` | 先启动 vla-body 再启动 vla-brain | 无 |
| `volumes` | 挂载宿主机目录 | `-v` / `--mount` |

`depends_on` 只控制**启动顺序**，不会等前一个容器里的程序准备就绪。

## command：body 为什么只启动 bash

```yaml
vla-body:
  command: ["bash"]
```

`command` 会覆盖镜像里默认的 CMD。vla-body 的启动命令就是 `bash`，容器起来后只运行一个 shell，**不会自动启动 body 服务**。文件开头的注释也写明了：

> Does not install host drivers or start body robot motion (body starts bash).

brain 不一样，它的 `command` 直接写了 `python scripts/serve_policy_rtc.py ...`，所以 `up -d` 后推理服务会自动起来。

### 实际启动 body 的流程

```bash
# 1. 启动两个容器：brain 自动起推理服务，body 只是待命
VLA_VERSION=2.1.0rc docker compose up -d

# 2. 进入 body 容器
docker exec -it vla-body-20261118 bash     # 或 docker compose exec vla-body bash

# 3. 在容器里手动启动 body 服务
python3 -m vla_body --mode runtime --joint-layout full-body \
    --control-mode servo --control-hz 30 --max-speed 1.0 \
    --action-index 0 --action-stride 1 --actions-per-inference 50 \
    --rtc-enabled --rtc-execution-horizon 10 --rtc-fixed-delay-steps 7 \
    --camera-stream-port 8090 \
    --prompt "Pick up two objects from the front box, one with each hand" \
    --gripper-threshold-enabled --camera-source shm --shm-top-roi 144 220 816 668
```

第 3 步的参数由负责人提供，会随任务变化，比如 `--prompt`。

### 为什么不让 body 自动启动

body 直接驱动电机，自动启动有安全风险：

- 配置里有 `restart: unless-stopped`。如果把 body 命令写进 `command`，机器断电重启或容器崩溃后，机器人会**自动开始动**，旁边可能有人。
- 手动启动时，操作人员可以先确认机器人周围安全、硬件状态正常，还可以按场景改参数。

如果确实要自动启动，可以把 `command` 改成上面的 python 命令，但要先和负责人确认安全问题，并建议把 `restart` 改成 `"no"`，同时给 body 加上 `logging` 限制（见下文）。

## stdin_open 和 tty

| 配置 | 等价参数 | 作用 |
|---|---|---|
| `stdin_open: true` | `-i` | 让容器的标准输入（stdin）保持打开 |
| `tty: true` | `-t` | 给容器分配一个虚拟终端 |

标准输入就是程序接收键盘输入的通道。bash 启动后会一直等着读输入：

- **没有** `stdin_open`：输入通道一开始就是关的，bash 一读就发现"不会再有输入了"，马上退出，容器也就停了。
- **有** `stdin_open: true`：输入通道一直开着，bash 一直等命令，容器保持运行。

`tty` 让你进去之后有命令提示符、颜色、方向键这些正常的终端体验。两个合起来就是常见的 `docker run -it`。

对 vla-body 来说，正是这两项让"只运行 bash"的容器能一直活着，你才能 `docker exec` 进去手动启动服务。

## ulimits.nofile：最大打开文件数

```yaml
ulimits:
  nofile:
    soft: 64000
    hard: 64000
```

`ulimits` 是 Linux 对进程的资源限制，`nofile` 指**一个进程最多能同时打开多少个文件**。

Linux 里"文件"的范围很广：普通文件、网络连接（socket）、设备（摄像头、串口）、管道，各占一个名额。很多系统默认上限只有 **1024**。

这个项目的程序会同时打开很多东西：

- brain：模型文件、大量 Python 库、网络连接。
- body：摄像头、电机设备、共享内存、和 brain 通信的 socket、日志文件。

超过上限时，程序报 `Too many open files` 然后崩溃，所以这里把上限调到 64000。

| | 含义 |
|---|---|
| `soft`（软限制） | 当前实际生效的上限 |
| `hard`（硬限制） | 天花板。进程可以自己把 soft 往上调，但不能超过 hard |

两个都设成 64000，意思是直接给到 64000，不用程序再调。

## logging：容器日志轮转

```yaml
logging:
  driver: json-file
  options:
    max-size: "20m"
    max-file: "5"
```

程序打印到屏幕的内容（stdout 和 stderr，比如 Python 的 `print` 和报错）会被 Docker 截下来存成文件。`docker compose logs vla-brain` 看到的就是这些。

| 配置 | 含义 |
|---|---|
| `driver: json-file` | 日志以 JSON 格式存在宿主机文件里，也是 Docker 的默认方式。位置在 `/var/lib/docker/containers/<容器ID>/<容器ID>-json.log` |
| `max-size: "20m"` | 一个日志文件最大 20 MB，写满就换新文件 |
| `max-file: "5"` | 最多保留 5 个文件，超过就删最旧的 |

所以 brain 的容器日志**最多占 20 MB × 5 = 100 MB**，只保留最近的内容。这叫日志轮转（log rotation）。

为什么要设：Docker 默认**不限制**日志大小。brain 是常驻的推理服务，跑几天、几周后日志可能涨到几十 GB，把 Jetson 磁盘写满，然后模型加载、写缓存都会失败。

两个注意点：

1. **vla-body 没有配 logging**，容器日志不限大小。目前 body 只运行 bash，输出很少；手动 `exec` 进去跑的 `python3 -m vla_body`，输出显示在你自己的终端里，不进容器日志，所以问题不大。以后如果改成自动启动 body，最好也加上。
2. logging 只管屏幕输出。body 挂载的 `body-logs` 目录是程序自己写的日志文件，不受 Docker 日志轮转管理，需要程序自己或其他工具清理。

## volumes：挂载了什么

代码已经打包在镜像里，**不挂载代码**。只挂载数据和硬件：

| 宿主机 | 容器 | 用途 | 谁用 |
|---|---|---|---|
| `/dev` | `/dev` | 相机、串口等设备 | 两个都挂 |
| `${VLA_DATA_ROOT}/.cache/openpi` | `/root/.cache/openpi` | tokenizer 缓存 | 两个都挂 |
| `${VLA_DATA_ROOT}/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace 缓存 | 两个都挂 |
| `${VLA_DATA_ROOT}/checkpoints` | `/workspace/checkpoints`（只读） | 模型权重 | brain |
| `${VLA_DATA_ROOT}/body-logs` | `/workspace/logs` | 日志和轨迹 | body |
| `/data/nvme/gino_motion_driver` | `/gino_motion_driver` | 电机驱动 | body |

### bind 是什么

`type: bind` 叫**绑定挂载**（bind mount）：把宿主机上的一个真实目录直接接到容器里的某个路径上。

可以理解成在容器里开了一扇"任意门"。容器里访问 `/gino_motion_driver`，实际读写的是宿主机的 `/data/nvme/gino_motion_driver`。两边是**同一份文件**，不是复制：

- 宿主机上改了文件，容器里立刻能看到。
- 容器删掉后，文件还留在宿主机上。

### 短写法和长写法

上表里的 6 条挂载**全都是 bind**，只是写法不同：

```yaml
# 短写法：宿主机路径:容器路径。冒号左边是路径，Docker 就认定是 bind
- ${VLA_DATA_ROOT:-/data/vla}/body-logs:/workspace/logs

# 长写法：同一件事拆开写，可以加更多选项
- type: bind
  source: /data/nvme/gino_motion_driver   # 宿主机路径（冒号左边）
  target: /gino_motion_driver             # 容器路径（冒号右边）
  bind:
    create_host_path: false               # 宿主机目录不存在时直接报错

- type: bind
  source: ${VLA_DATA_ROOT:-/data/vla}/checkpoints
  target: /workspace/checkpoints
  read_only: true                         # 容器里只读，防止误改权重
  bind:
    create_host_path: false
```

用长写法的关键是 `create_host_path: false`，短写法加不了这个选项：

- **短写法**：宿主机目录不存在时，Docker **自动建一个空目录**挂进去，不报错。容器能启动，但程序运行时才发现找不到文件，问题很难查。
- **`create_host_path: false`**：目录不存在时，容器**直接启动失败并报错**，问题在启动时就暴露。

电机驱动和模型权重是必须事先放好的，所以用了长写法。cache 和 logs 目录一开始是空的也没关系，用短写法让 Docker 自动创建即可。

### 另一种挂载：命名卷

除了 bind，Docker 还有 `type: volume`（命名卷），由 Docker 自己管理存储位置，你不需要知道文件在宿主机哪里。这个文件没用到，因为模型、驱动、日志都需要放在宿主机上固定、可见的位置。

## compose 和 Dockerfile 怎么配合

| | Dockerfile.release 的默认值 | docker-compose.yml |
|---|---|---|
| ENTRYPOINT | `nvidia_entrypoint.sh` | 不覆盖，保留 |
| CMD | brain 是 `serve_policy_rtc.py --help`，body 是 `bash` | brain 换成真正的启动命令，body 仍是 `bash` |
| 挂载点 | `mkdir` 了 `/workspace/checkpoints`、`/workspace/logs` | 把宿主机目录挂上去 |
| 工作目录 | `WORKDIR /workspace` | `--dir checkpoints/...` 的相对路径基于它 |

brain 实际执行的命令：

```
/opt/nvidia/nvidia_entrypoint.sh python scripts/serve_policy_rtc.py \
    --config pi05_gino_cemat_1031_action_zishiying_cut_2x \
    --dir checkpoints/pi05_gino_cemat_1031_action_zishiying_cut_2x_pytorch \
    --port 8000
```

- `--config` 和 `--dir` 必须配套。权重目录里要有 `model.safetensors` 和该 config 对应的 `assets/<asset_id>/norm_stats.json`，否则启动时报 "Norm stats file not found"。
- 两个容器都用 host 网络，body 通过 `localhost:8000` 连接 brain。
