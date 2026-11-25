# Docker 与 Dockerfile 系统学习：以 brain-service/docker-deploy 为教材

面向**会用 `docker run` 启动现成镜像、但没自己写过 Dockerfile 的读者**。
这个教程不从 "hello world" 讲起，而是拿 `/root/brain/brain-service/docker-deploy`
这套真实在用的部署代码当教材：每个概念都先指出它在仓库里出现在哪一行，再讲原理，
最后给一个可以自己动手跑的小实验。

仓库里写得好的地方会讲"为什么这么写"，写得有问题的地方（包括几个真 bug）
也会明确指出 —— 读别人的 Dockerfile 时能看出问题，才算真的学会了。

## 关于验证（请先读）

`code/` 下 7 个实验**全部在真实机器上跑过**，各实验 README 里的数字都是实测值：

```
机器      : NVIDIA Jetson Thor (aarch64), L4T R38.4.0
Docker    : 29.1.4（BuildKit v0.26.3）
buildx    : v0.30.1
Compose   : v5.0.1
存储驱动  : fuse-overlayfs；默认 runtime: nvidia
```

正文里对仓库行为的判断（ARG 作用域、compose 引号、PID 1 信号、`--target` 不存在、`docker save ""` 等）
也都在这台机器上做了最小复现。仍然无法在这里验证的少数几处（需要真实的 Harbor/ECR、bazel 构建），
标了 **【待验证】**。

实验里**推翻了我事先预测**的结果有一个，已在第 4、6 章和 lab06 中标出：
compose 里借用 `postgres:16` 镜像时，`init: true` 并没有让 `docker stop` 变快 ——
因为这个镜像自带 `STOPSIGNAL SIGINT`，而 bash 对 SIGINT 有特殊处理。

## 学习路线

| 章节 | 主题 | 主要对照的仓库文件 |
|---|---|---|
| [00_overview.md](00_overview.md) | 心智模型：镜像/层/容器/仓库，这套仓库的完整生命周期 | 全部 |
| [01_dockerfile_instructions.md](01_dockerfile_instructions.md) | Dockerfile 指令逐条精讲 | `brain-service/Dockerfile_*`、`robot-data-platform/Dockerfile`、`model-bin/Dockerfile` |
| [02_context_and_cache.md](02_context_and_cache.md) | 构建上下文、`.dockerignore`、层缓存、镜像体积 | `build_docker.sh`、`infra-bin/build.sh` |
| [03_multistage_and_targets.md](03_multistage_and_targets.md) | 多阶段构建、`--target`、BuildKit 新语法 | `covariant/infra/video_monitoring/local_clipping/Dockerfile` |
| [04_entrypoint_cmd_signals.md](04_entrypoint_cmd_signals.md) | ENTRYPOINT/CMD、exec 形式、PID 1 与信号 | `entrypoint.sh`、`run_brain.sh`、`docker_entrypoint.py` |
| [05_docker_run_in_depth.md](05_docker_run_in_depth.md) | `docker run` 参数全解：网络/IPC/设备/GPU/挂载/环境变量 | `brain-service/scripts/dev.sh` |
| [06_compose.md](06_compose.md) | Docker Compose：多服务编排、secrets、网络 | `universal-uploader/assert/docker-compose.yaml`、`deploy.sh`、`video-monitoring/*` |
| [07_tag_push_multiarch.md](07_tag_push_multiarch.md) | 命名与 tag、私有仓库、save/load 离线分发、多架构 manifest | 各 `build.sh` 的 push / `--create-manifest` 段 |
| [08_review_and_best_practices.md](08_review_and_best_practices.md) | 仓库问题清单 + 最佳实践 + 一份改写后的 Dockerfile | 全部 |
| [09_cheatsheet.md](09_cheatsheet.md) | 命令速查 | — |
| [code/](code/) | 7 个动手实验 | — |

建议顺序：00 → 01 → 02 → 做 lab01/lab02 → 03 → lab03 → 04 → lab04 → 05 → 06 → lab06 → 07 → 08。

## 如果只看一段：从这套仓库里能学到的 8 件事

1. **镜像 = 只读的层叠文件系统 + 一份元数据（默认命令、环境变量、工作目录）。**
   Dockerfile 里每条 `RUN/COPY/ADD` 生成一层，其他指令只改元数据。（第 0、1 章）
2. **构建上下文决定了 `COPY` 能看到什么。** `brain-service` 的构建上下文是整个
   `/root/brain`（仓库的上一级目录），且没有 `.dockerignore`，会把 `.git` 等全部发给 daemon；
   `infra-bin` 则特意 `cd bazel-bin` 再 build，并用 `tar -h` 把 bazel 的软链接实体化 ——
   因为 **Docker 不会跟随指向上下文之外的软链接**。（第 2 章）
3. **层缓存是按顺序失效的。** `robot-data-platform` 先 COPY 依赖清单再装依赖、最后 COPY 源码，
   是标准写法；`brain-service` 把整个源码 COPY 进去再 `bazel build`，任何一个文件改动都会
   让后面所有层重建。（第 2 章）
4. **多阶段构建 + `--target` 可以一个 Dockerfile 产出 dev/test/production 多个镜像。**
   `local_clipping/Dockerfile` 是仓库里的范本。而 `build_docker.sh --dev` 用的
   `--target covariant_dev` 在 Dockerfile 里根本不存在。（第 3 章）
5. **`ENTRYPOINT` 脚本最后一行要用 `exec`。** `entrypoint.sh` 写对了，`run_brain.sh` 没写，
   后果是 `docker stop` 要等 10 秒超时后强杀，服务收不到 SIGTERM，无法优雅退出。（第 4 章）
6. **`dev.sh` 的 `docker run` 是一份很好的"参数大全"**：`--network host`、`--ipc host`、
   `--privileged`、`-v /dev:/dev`、`--gpus all`、挂载 docker.sock、用宿主机源码覆盖镜像内源码……
   每一个都有明确的理由和代价。（第 5 章）
7. **Compose 的 `depends_on` 只保证启动顺序，不保证"依赖已就绪"**；`deploy.sh` 里的
   `docker-compose down -v` 会删除命名卷，目前没出事只是因为数据库用的是 bind mount。（第 6 章）
8. **多架构镜像靠 manifest list。** 仓库用"两台机器分别 build + `docker manifest create`"的方式；
   但 `build_docker.sh` 默认 tag 是 `arm64_latest`（架构作前缀），manifest 逻辑却按
   `_aarch64` 后缀解析，两套命名约定是冲突的。（第 7 章）

## 仓库文件地图

```
docker-deploy/
├── README.md                 # 打包→拷贝→docker load→compose up 的流程（第 7 章）
├── env.sh                    # 所有脚本依赖的环境变量（第 5、6 章）
├── brain-service/            # 主服务镜像
│   ├── Dockerfile_x86_64     # 两个 Dockerfile 只差 bazel build 的一个 -s 参数（第 1、8 章）
│   ├── Dockerfile_aarch64
│   ├── assets/               # requirements、ueye 相机配置生成脚本
│   └── scripts/
│       ├── build_docker.sh   # build / push / export / manifest 一条龙（第 2、7 章）
│       ├── dev.sh            # 开发容器启动脚本（第 5 章，重点）
│       ├── run_brain.sh      # 生产镜像 ENTRYPOINT（第 4 章）
│       └── build_singularity.sh  # Docker 镜像 → Singularity/Apptainer 镜像（第 7 章）
├── infra-bin/ model-bin/     # "基础镜像 + bazel 产物" 的薄镜像（第 1、2、4 章）
├── robot-data-platform/      # Django + gunicorn，最"教科书"的 Dockerfile（第 1、2 章）
├── universal-uploader/       # 上传服务 + Postgres，compose 编排（第 6 章，重点）
│   └── event-uploader/       # python-slim 小镜像
└── video-monitoring/         # 多个 compose 模板、Makefile 驱动的多阶段构建（第 3、6 章）
```
