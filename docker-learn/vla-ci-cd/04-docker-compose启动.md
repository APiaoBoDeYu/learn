# 04 docker-compose.yml：在机器人上启动容器

文件：`embodied_algo/body/gino/docker-compose/docker-compose.yml`

## docker-compose 是什么

`docker run` 一次启动一个容器，参数都写在命令行上，又长又容易写错。docker-compose 把一组容器的启动参数写进一个 YAML 文件，用一条命令统一管理：

```bash
docker compose up -d        # 拉取镜像（本地没有时）并在后台启动所有服务
docker compose pull         # 拉取最新镜像（tag 被覆盖过时需要）
docker compose ps           # 查看状态
docker compose logs -f vla-brain
docker compose exec vla-body bash
docker compose down         # 停止并删除容器
```

文件里的每个 `services` 条目对应一个容器。这里有两个：`vla-body` 和 `vla-brain`。

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

## 逐项说明

以 `vla-brain` 为例，`vla-body` 大部分相同。

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
| `stdin_open` + `tty` | 保持标准输入并分配终端，方便 `docker attach` | `-it` |
| `restart: unless-stopped` | 容器退出或机器重启后自动拉起，手动 stop 的除外 | `--restart unless-stopped` |
| `ulimits.nofile` | 最大打开文件数调到 64000 | `--ulimit nofile=64000:64000` |
| `logging` | 容器日志单文件最大 20 MB，保留 5 个，防止占满磁盘 | `--log-opt` |
| `depends_on` | 先启动 vla-body 再启动 vla-brain | 无 |
| `volumes` | 挂载宿主机目录 | `-v` |

`depends_on` 只控制**启动顺序**，不会等前一个容器里的程序准备就绪。

## 挂载了什么

代码已经打包在镜像里，**不挂载代码**。只挂载数据和硬件：

| 宿主机 | 容器 | 用途 | 谁用 |
|---|---|---|---|
| `/dev` | `/dev` | 相机、串口等设备 | 两个都挂 |
| `${VLA_DATA_ROOT}/.cache/openpi` | `/root/.cache/openpi` | tokenizer 缓存 | 两个都挂 |
| `${VLA_DATA_ROOT}/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace 缓存 | 两个都挂 |
| `${VLA_DATA_ROOT}/checkpoints` | `/workspace/checkpoints`（只读） | 模型权重 | brain |
| `${VLA_DATA_ROOT}/body-logs` | `/workspace/logs` | 日志和轨迹 | body |
| `/data/nvme/gino_motion_driver` | `/gino_motion_driver` | 电机驱动 | body |

挂载有两种写法：

```yaml
# 短写法：宿主机:容器
- ${VLA_DATA_ROOT:-/data/vla}/body-logs:/workspace/logs

# 长写法：能设置更多选项
- type: bind
  source: ${VLA_DATA_ROOT:-/data/vla}/checkpoints
  target: /workspace/checkpoints
  read_only: true                  # 容器里只读，防止误改权重
  bind:
    create_host_path: false        # 宿主机目录不存在时直接报错
```

短写法遇到宿主机目录不存在时，会自动创建一个空目录，容器能启动，但程序找不到文件，问题到运行时才暴露。`create_host_path: false` 让问题在启动时就暴露出来，所以权重和电机驱动这两个必须存在的目录用了长写法。

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

body 启动后只是一个 bash，不会自动让机器人动起来。需要 `docker compose exec vla-body bash` 进去手动启动程序。
