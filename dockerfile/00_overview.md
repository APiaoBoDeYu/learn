# 第 0 章：心智模型 —— 你每天用的 `docker run` 背后是什么

你已经会 `docker pull` / `docker run` / `docker exec`。这一章把这些动作背后的几个对象讲清楚，
后面所有章节都建立在这几个概念上。

## 0.1 五个核心对象

```
            docker build                docker run
Dockerfile ─────────────▶  镜像 Image  ─────────────▶  容器 Container
 + 构建上下文                 │   ▲                      (镜像 + 一层可写层 + 进程)
                             │   │
                 docker push │   │ docker pull
                             ▼   │
                         仓库 Registry
                 (Docker Hub / AWS ECR / Harbor)
```

| 对象 | 是什么 | 类比 |
|---|---|---|
| **Dockerfile** | 一份"怎么造镜像"的脚本 | 源代码 |
| **构建上下文 (build context)** | `docker build` 最后那个路径参数所指的目录，会被整个打包发给 daemon | 编译时能看到的源文件集合 |
| **镜像 (image)** | 一组只读的文件系统层 + 一份元数据（JSON） | 编译产物 / 可执行文件 |
| **容器 (container)** | 镜像之上加一层可写层，再在隔离环境里跑一个进程 | 运行中的进程 |
| **仓库 (registry)** | 存放镜像的服务器 | 包管理仓库（PyPI） |

最容易误解的一点：**容器不是虚拟机**。容器就是宿主机上一个普通进程，只是被内核的
namespace（看不到别的进程、网络、挂载点）和 cgroup（限制 CPU/内存）圈了起来。
所以 `dev.sh` 里 `--network host`、`--ipc host`、`--pid host`（compose 里）这类参数的含义就是
"这一项不隔离了，直接用宿主机的"。第 5 章细讲。

## 0.2 镜像是"层"叠出来的

看仓库里最简单的一个 Dockerfile，`docker-deploy/model-bin/Dockerfile`：

```dockerfile
ARG BASE_IMAGE
ARG BAZEL_BIN
FROM ${BASE_IMAGE}                       # 从已有镜像的全部层开始
RUN mkdir /model-bin                     # 层 +1：新增一个空目录
WORKDIR /model-bin                       # 只改元数据（默认工作目录）
COPY ${BAZEL_BIN} bazel-bin              # 层 +1：bazel 产物
WORKDIR /model-bin/bazel-bin/covariant/model   # 只改元数据
COPY entrypoint.sh entrypoint.sh         # 层 +1
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]      # 只改元数据（默认启动命令）
```

结果镜像 = `brain-service-dev` 的所有层 + 3 层新层 + 一份元数据。

```
┌───────────────────────────────┐  ← 容器运行时的可写层（容器删了就没了）
├───────────────────────────────┤
│ COPY entrypoint.sh            │  ┐
│ COPY bazel-bin                │  │ model-bin 自己新增的层
│ RUN mkdir /model-bin          │  ┘
├───────────────────────────────┤
│ brain-service-dev 的若干层     │  ┐ 基础镜像，多个镜像共享，
│ ...                           │  │ 磁盘上只存一份
│ ubuntu 基础层                  │  ┘
└───────────────────────────────┘
```

层带来的三个直接后果，后面章节会反复用到：

1. **共享**：`infra-bin` 和 `model-bin` 都 `FROM brain-service-dev`，基础层在磁盘上只存一份，
   push/pull 时也只传差异层。
2. **缓存**：重新 build 时，某一层的输入没变就直接复用。理解缓存失效规则是写好 Dockerfile 的关键（第 2 章）。
3. **只增不减**：上层删除下层的文件，只是"盖住"，镜像不会变小。所以
   `apt-get install ... && rm -rf /var/lib/apt/lists/*` 必须写在**同一个 RUN** 里（第 2 章）。

可以用这两个命令自己看层：

```bash
docker history <镜像名>            # 每层由哪条指令产生、多大
docker image inspect <镜像名>      # 元数据：Env、Entrypoint、Cmd、WorkingDir、Layers
```

## 0.3 容器 = 镜像 + 可写层 + 一个进程

`docker run` 做了这几件事：

1. 在镜像顶上加一层可写层（容器里所有写操作都落在这里）。
2. 按参数准备隔离环境：网络、挂载卷、设备、环境变量……
3. 启动**一个**进程：`ENTRYPOINT + CMD`（可被 `docker run` 参数覆盖）。这个进程在容器里是 PID 1。
4. **这个进程退出，容器就停止。** 这就是为什么 compose 里 `uploader_finalize_orphaned_files`
   要写成 `while true; do ...; sleep 120; done` —— 进程必须一直活着。

可写层的生命周期 = 容器的生命周期。`dev.sh` 用了 `--rm`，退出即删除容器，可写层里的改动全丢；
需要保留的东西（源码、bazel 缓存、数据）都通过 `-v` 挂载到宿主机目录。

## 0.4 这套仓库的完整生命周期

把 docker-deploy 里所有脚本串起来，就是一条典型的"镜像供应链"：

```
                    ┌──────────────────────────────────────┐
  (仓库外的某处)     │ brain-service-dev:v1.1.0              │  开发镜像：ubuntu + CUDA + bazel
  build_docker.sh   │ (bazel、node_modules、third_party      │  + node + ueye 驱动 + /opt/geekplus/*
  --dev             │  都预装在 /opt/geekplus 下)            │
                    └──────┬───────────────┬────────────────┘
                           │ FROM          │ FROM
      ┌────────────────────┘               └───────────────┐
      ▼                                                    ▼
 brain-service 镜像                               infra-bin / model-bin 镜像
 Dockerfile_${arch}:                              宿主机先 bazel build，
 COPY 源码 → 镜像内 bazel build                    再把 bazel-bin COPY 进去
      │
      ├── docker push → AWS ECR + Harbor (robot-harbor.geekplus.com/gino)
      ├── docker manifest create → 把 x86_64 + aarch64 合成一个多架构 tag
      ├── docker save → .tar → 拷到工作站 → docker load   （离线部署）
      └── singularity build → .simg                         （HPC/集群）

 工作站上：
   dev.sh           → docker run 一个开发容器，挂载宿主机源码
   deploy.sh        → 检查目录/secrets → docker-compose up -d（uploader + postgres）
   video-monitoring → 多个 compose 模板
```

两种不同的"把代码装进镜像"的策略，在这里同时存在：

| 策略 | 例子 | 优点 | 缺点 |
|---|---|---|---|
| **在镜像里编译** | `brain-service/Dockerfile_*`：COPY 源码 → RUN bazel build | 构建环境可复现，宿主机不需要任何工具链 | 镜像里带着完整源码和工具链，体积大；缓存难利用 |
| **在宿主机编译，镜像只装产物** | `infra-bin`、`model-bin`、`universal-uploader` | 镜像小、构建快 | 依赖宿主机环境（这里实际是"在 dev 容器里"跑 build.sh） |

第三种更好的做法是**多阶段构建**：在第一个阶段编译、第二个阶段只拷产物，兼得两者优点。
仓库里 `local_clipping/Dockerfile` 用了这个思路，第 3 章讲。

## 0.5 Docker 的几个组件名词

后面会碰到，先混个脸熟：

| 名词 | 说明 |
|---|---|
| **Docker CLI** | 你敲的 `docker` 命令，本身只是客户端 |
| **dockerd (daemon)** | 真正干活的后台服务，CLI 通过 `/var/run/docker.sock` 跟它通信。`dev.sh` 把这个 socket 挂进容器，容器里就能操作宿主机的 docker（第 5 章） |
| **BuildKit** | 新一代构建引擎，Docker 23+ 默认启用。支持并行构建、`--mount=type=cache`、`# syntax=` 等。 |
| **buildx** | BuildKit 的 CLI 插件，用于多平台构建 `--platform linux/amd64,linux/arm64`（第 7 章） |
| **containerd / runc** | dockerd 下面真正创建容器的底层运行时 |
| **nvidia-container-toolkit** | 让容器能用 GPU 的组件，`--gpus all` / `--runtime nvidia` 依赖它 |
| **Compose** | 用 YAML 描述多个容器。v1 是独立的 `docker-compose`（Python 写的，已停止维护），v2 是 `docker compose`（CLI 插件）。仓库里两种写法都有（第 6 章） |
| **OCI** | 镜像格式和运行时的开放标准。Singularity/Apptainer、Podman 都能用 OCI 镜像 |

## 0.6 检查自己理解了没有

- `docker stop` 一个容器后再 `docker start`，容器里之前写的文件还在吗？（在。可写层只在 `docker rm` 时删除）
- `dev.sh` 用了 `--rm`，在容器里 `pip install` 了一个包，退出再进，包还在吗？（不在）
- 为什么 `infra-bin` 和 `model-bin` 都 `FROM brain-service-dev`，但 push 时基础层只上传一次？（层按内容哈希寻址，仓库里已有的层不重复传）

下一章：[01_dockerfile_instructions.md](01_dockerfile_instructions.md)
