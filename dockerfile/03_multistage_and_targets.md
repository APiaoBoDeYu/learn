# 第 3 章：多阶段构建、`--target` 与 BuildKit 新语法

## 3.1 问题：编译环境 ≠ 运行环境

第 2 章看到，brain-service 生产镜像里装着 bazel、node、gcc、完整源码、构建缓存 ——
这些东西**运行时一个都用不上**。仓库用两种办法回避这个问题：

- infra-bin/model-bin：在宿主机（实际是 dev 容器）里先编译，镜像只 COPY 产物。缺点是构建流程分散在 shell 脚本里，依赖执行环境。
- 直接接受大镜像（brain-service）。

多阶段构建（multi-stage build）是更干净的方案：**一个 Dockerfile 里写多个 `FROM`，
每个 `FROM` 开启一个阶段，后面的阶段可以从前面的阶段 `COPY --from` 文件。最终镜像只包含最后一个（或 `--target` 指定的）阶段。**

```dockerfile
FROM gcc:13 AS builder               # 阶段 1：有完整编译器
COPY hello.c .
RUN gcc -O2 -static -o hello hello.c

FROM debian:bookworm-slim            # 阶段 2：干净的运行环境
COPY --from=builder /hello /usr/local/bin/hello    # 只拿产物
CMD ["hello"]
```

builder 阶段的 gcc、源码、中间文件**都不会**出现在最终镜像里。

## 3.2 仓库范本：`local_clipping/Dockerfile`

文件在 `covariant/infra/video_monitoring/local_clipping/Dockerfile`，由
`docker-deploy/video-monitoring/build.sh` → `make prod` → `DOCKER_TARGET=production` 驱动。
它把**一个 Dockerfile 产出 dev / test / production 三种镜像**做得很漂亮：

```dockerfile
# syntax = robot-harbor.geekplus.com/gino/dockerfile:1.7-labs
FROM robot-harbor.geekplus.com/gino/python:3.11-slim AS base

ARG FFMPEG_VERSION=6.0
ARG FFMPEG_ARTIFACT_NAME=ffmpeg-$FFMPEG_VERSION-amd64-static
ENV INSTANCE_NAME="video_monitoring.local_clipping"
ENV PATH="/usr/local/bin:${PATH}"

RUN apt-get update && apt-get upgrade -y && apt-get install -q -y gcc && \
    apt-get install -q -y --no-install-recommends curl wget xz-utils && \
    rm -rf /var/lib/apt/lists/*
RUN wget -O /tmp/$FFMPEG_ARTIFACT_NAME.tar.xz https://.../$FFMPEG_ARTIFACT_NAME.tar.xz && \
    tar xvf ... && mv .../ffmpeg .../ffprobe /usr/local/bin/ && rm ... && rm -r ...
WORKDIR /root
COPY deps/ /root/deps
RUN pip install --no-cache-dir deps/cov_libs && rm -rf /root/deps

# The development docker image.
FROM base AS dev
COPY requirements/ /root/requirements/
RUN pip install --no-cache-dir -r requirements/test.txt
WORKDIR /root/code
ENV PYTHONPATH="/root/code"

# The test docker image.
FROM dev AS test
COPY --exclude=deps . /root/code
ENTRYPOINT ["/bin/bash", "test_all.sh"]

# The production docker image.
FROM base AS production
COPY requirements/ /root/requirements/
RUN pip install --no-cache-dir -r requirements/base.txt && \
    rm -r requirements && \
    apt-get purge -q -y --auto-remove gcc xz-utils
WORKDIR /opt/local_clipping
COPY server/ server/
COPY client/ client/
COPY telemetry_clipping/ telemetry_clipping/
COPY launch.py launch.py
ENTRYPOINT ["python", "launch.py"]
```

阶段关系图：

```
            base  (python-slim + ffmpeg + cov_libs)
           ╱    ╲
        dev      production   ← build --target production：只构建 base → production
         │                       dev/test 阶段完全跳过（BuildKit 只构建目标阶段依赖的阶段）
        test
```

| 阶段 | 用途 | 关键点 |
|---|---|---|
| `base` | 公共部分 | 最慢的 apt、ffmpeg 下载、cov_libs 编译只写一次 |
| `dev` | 开发容器 | 装 **test** 依赖（含 pytest 等），**不 COPY 源码**——源码运行时用 `-v $(WORKING_DIR):/root/code` 挂进来（见 Makefile 的 `DEV_VOLUME_MOUNTS`） |
| `test` | CI 跑测试 | 在 dev 基础上把源码 COPY 进去，ENTRYPOINT 直接跑测试 |
| `production` | 部署 | 只装 **base** 依赖，装完立即卸载 gcc，只拷运行需要的四个目录 |

`--target` 的用法：

```bash
docker build --target production -t video-local-clipping .
docker build --target dev        -t video-local-clipping-dev .
```

不指定 `--target` 时默认构建**最后一个**阶段（这里恰好是 production）。

### 这个文件值得学的几个细节

1. **dev 阶段不 COPY 源码**。开发时源码天天变，放进镜像意味着每次都要重建；
   用 bind mount 挂进来，改完立即生效。这和 `brain-service/scripts/dev.sh` 的思路一致（第 5 章）。
2. **`COPY --exclude=deps . /root/code`**：把整个上下文拷进来但排除 `deps/`。
   这是 Dockerfile 1.7 **labs**（实验）语法，所以文件第一行必须写 `# syntax = .../dockerfile:1.7-labs`。
   较新的 Dockerfile 前端已把 `--exclude` 转为稳定语法：实测在 Docker 29.1.4（BuildKit v0.26.3）上，
   写 `# syntax=docker/dockerfile:1` 或者干脆不写 syntax 行，`COPY --exclude=deps . /code` 都能正常工作。
   仓库里的 `1.7-labs` 是在旧版 Docker 上的必要写法。
3. **production 和 dev 都从 base 分叉**，而不是 production 从 dev 继承 —— 测试依赖不会混进生产镜像。

### 这个文件的问题

1. **`ARG FFMPEG_ARTIFACT_NAME=ffmpeg-$FFMPEG_VERSION-amd64-static`：架构写死了 amd64。**
   这个镜像在 aarch64（Jetson）上构建时，下载的仍是 x86 的 ffmpeg 二进制，运行时会报
   `exec format error`。而 `video-monitoring/build.sh` 明确支持 `--create-manifest` 合并 x86_64 和 aarch64 两个镜像。
   修法是用 BuildKit 自动提供的 `TARGETARCH`：

   ```dockerfile
   ARG TARGETARCH                                  # BuildKit 自动注入：amd64 / arm64
   ARG FFMPEG_ARTIFACT_NAME=ffmpeg-${FFMPEG_VERSION}-${TARGETARCH}-static
   ```

   （johnvansickle.com 的 ffmpeg 静态包恰好用 `amd64` / `arm64` 命名，与 TARGETARCH 一致。）
2. **`apt-get upgrade -y`**：会让构建结果依赖"构建当天的仓库状态"，不可复现。官方建议不在 Dockerfile 里做 upgrade，
   而是定期更新基础镜像 tag。这是一个有争议的点（安全补丁 vs 可复现），了解取舍即可。
3. **gcc 装在 base 阶段**，production 阶段通过 `apt-get purge` 卸载。卸载有效（同一个 RUN），
   但 base 阶段那层里 gcc 依然存在——**purge 只是在新层里标记删除**，production 镜像体积并不会因此减小。
   真正省体积的写法是再加一个 builder 阶段：

   ```dockerfile
   FROM base AS builder
   RUN apt-get update && apt-get install -y gcc && pip wheel --wheel-dir=/wheels -r requirements/base.txt

   FROM python:3.11-slim AS production        # 不从 base 继承 gcc
   COPY --from=builder /wheels /wheels
   RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
   ```

## 3.3 仓库里的 `--target` 误用

`brain-service/scripts/build_docker.sh`：

```bash
build_target="brain-service"
...
--dev)
    IMAGE="brain-service-dev"
    build_target="covariant_dev"
...
docker build ... --target "$build_target" ...
```

但 `Dockerfile_x86_64` / `Dockerfile_aarch64` 里只有一个阶段：

```dockerfile
FROM robot-harbor.geekplus.com/gino/brain-service-dev:v1.1.0 as brain-service
```

**根本不存在 `covariant_dev` 这个阶段**，所以 README 里写的 `build_docker.sh --dev` 会直接失败：

```
ERROR: failed to build: failed to solve: target stage "covariant_dev" could not be found
```

（这是用一个只有 `FROM busybox AS brain-service` 的 Dockerfile 加 `--target covariant_dev` 实测得到的报错原文。）

推测原因：dev 镜像原本和生产镜像写在同一个多阶段 Dockerfile 里（`covariant_dev` 阶段在前，
`brain-service` 阶段 `FROM covariant_dev`），后来 dev 镜像的定义被拆走、改为直接引用
Harbor 上的 `brain-service-dev:v1.1.0`，但脚本没同步更新。
**dev 镜像本身的 Dockerfile 不在这个仓库里**——如果要改 dev 镜像（比如升级 CUDA），需要先找到它的来源。

## 3.4 `COPY --from` 的三种用法

```dockerfile
COPY --from=builder /out/app /usr/local/bin/app          # ① 从本 Dockerfile 的某个阶段
COPY --from=nginx:1.27 /etc/nginx/nginx.conf /tmp/        # ② 直接从任意镜像拷文件
COPY --from=shared . /app/shared                          # ③ 从命名上下文（--build-context shared=...，第 2 章）
```

用法 ② 很实用：比如从官方 `docker:cli` 镜像里只拿 `docker` 二进制，不用 apt 装：

```dockerfile
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
```

## 3.5 BuildKit 的 `RUN --mount`

BuildKit 给 `RUN` 加了挂载能力，挂进来的东西**只在这条 RUN 执行期间存在，不进镜像层**。

### `type=cache`：跨构建持久化的缓存目录

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

pip/apt/bazel/yarn 的下载缓存保存在 BuildKit 自己管理的缓存里，下次构建即使这层缓存失效，
也不用重新从网络下载。注意此时**不要**再加 `--no-cache-dir`，否则缓存挂载就没意义了。

apt 的写法稍复杂（Debian 官方镜像默认配置了"装完自动清缓存"，要先关掉）：

```dockerfile
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends gcc
```

### `type=bind`：只读挂载上下文里的文件，不产生 COPY 层

```dockerfile
RUN --mount=type=bind,source=requirements.txt,target=/tmp/req.txt \
    pip install -r /tmp/req.txt
```

### `type=secret`：构建时用密钥，又不留在镜像里

```bash
docker build --secret id=aws,src=$HOME/.aws/credentials .
```

```dockerfile
RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
    aws s3 cp s3://bucket/model.bin /opt/model.bin
```

对比 `ARG AWS_SECRET_ACCESS_KEY`：ARG 的值会出现在 `docker history` 里，secret 挂载不会。

### `type=ssh`：构建时用宿主机的 ssh-agent 拉私有 git 仓库

```bash
docker build --ssh default .
```

```dockerfile
RUN --mount=type=ssh git clone git@github.com:org/private.git
```

## 3.6 ⚠ 多阶段构建与 bazel runfiles

如果你打算把 brain-service 改成多阶段构建，要注意 bazel 产物的特殊性：
`bazel-bin/covariant/service/launch` 是个启动脚本，它依赖旁边的 `launch.runfiles/` 目录，
而 runfiles 目录里几乎全是**指向 output base 的软链接**。`COPY --from=builder` 同样不跟随软链接，
所以要在 builder 阶段先实体化（这正是 infra-bin 用 `tar -h` 的原因）：

```dockerfile
FROM brain-service-dev:v1.1.0 AS builder
...
RUN bazel build //covariant/service:launch && \
    mkdir -p /out && cp -rL bazel-bin/covariant/service/launch bazel-bin/covariant/service/launch.runfiles /out/

FROM <运行时基础镜像> AS runtime
COPY --from=builder /out /opt/brain
```

另外 brain-service 运行时还依赖 ueye 相机驱动、CUDA 运行库等，运行时基础镜像不能随便选个 slim 镜像，
需要先梳理清楚运行时依赖。这正是很多团队"干脆直接用 dev 镜像当基础"的原因 —— 简单，但大。

## 3.7 动手

做 [code/lab03_multistage](code/lab03_multistage/)：同一个程序，单阶段 vs 多阶段，对比镜像体积；
再用 `--target` 构建出 dev 和 production 两个镜像。

下一章：[04_entrypoint_cmd_signals.md](04_entrypoint_cmd_signals.md)
