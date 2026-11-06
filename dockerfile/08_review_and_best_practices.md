# 第 8 章：仓库问题清单、最佳实践与改写示例

前面各章零散指出了很多问题，这里汇总成一张表，并给出最佳实践清单和改写后的 brain-service Dockerfile。

## 8.1 问题清单（按影响排序）

"确认"表示从代码逻辑可以直接确定；"实测"表示在 Jetson Thor（Docker 29.1.4）上用最小复现跑过；"待验证"表示需要真实的 bazel/Harbor 环境。

| # | 文件 | 问题 | 后果 | 状态 | 章节 |
|---|---|---|---|---|---|
| 1 | `brain-service/scripts/run_brain.sh` | 启动 launch 时没有 `exec`；bash 当 PID 1 | `docker stop` 时服务收不到 SIGTERM，10 秒后被 SIGKILL，无法优雅关闭硬件/连接 | 实测（lab04 机制复现：10.9s / 137） | 4.3 |
| 2 | `brain-service/scripts/build_docker.sh` | `--export` 用了未定义的 `$tag` | README 描述的离线部署流程第一步就失败 | 实测（`invalid reference format`） | 7.3 |
| 3 | `brain-service/scripts/build_docker.sh` | `--dev` 使用 `--target covariant_dev`，Dockerfile 中不存在该阶段 | README 描述的 dev 镜像构建失败 | 实测（`target stage "covariant_dev" could not be found`） | 3.3 |
| 4 | `universal-uploader/deploy.sh` | 部署时 `docker-compose down -v` | 目前无害（db 用 bind mount）；一旦改成命名卷，每次部署都会清空数据库 | 实测（lab06） | 6.4 |
| 5 | `universal-uploader/assert/docker-compose.yaml` | `- MANAGED_DATA_S3_BUCKET="..."` 列表写法带引号 | 变量值里包含字面量双引号 | 实测（lab06：`'"demo-internal-data-us"'`） | 6.3 |
| 6 | 同上 | db `ports: "6432:6432"` 绑定在 0.0.0.0 | 数据库对局域网开放 | 确认 | 6.3 |
| 7 | 同上 | `depends_on` 没有健康检查 | 首次启动 uploader 可能多次崩溃重启，靠 `restart: always` 兜底 | 实测（lab06：`Connection refused`） | 6.3 |
| 8 | 同上 | finalize 服务没有 `restart` 策略 | 宿主机重启后清理任务不会恢复 | 确认 | 6.3 |
| 9 | `local_clipping/Dockerfile` | ffmpeg 下载地址写死 `amd64` | 在 aarch64 上构建出的镜像里 ffmpeg 无法执行 | 确认 | 3.2 |
| 10 | `model-bin/`、`infra-bin/Dockerfile` | `ARG BAZEL_BIN` 声明在 `FROM` 前，阶段内未重新声明 | `${BAZEL_BIN}` 为空，`--build-arg` 无效；目前碰巧能工作 | 实测（lab05：`BAZEL_BIN=''`、`UndefinedVar`） | 1.6 |
| 11 | `brain-service/Dockerfile_*` | `GIT_COMMIT` 等 ARG 从未被传入 | 镜像里版本信息为空/0 | 确认 | 1.6 |
| 12 | `brain-service/Dockerfile_*` | 先 COPY 全部源码再做与源码无关的 `cp -r`；生产镜像含源码、工具链、bazel 缓存、重复的 third_party | 构建慢、镜像大 | 确认 | 2.2 |
| 13 | `brain-service/Dockerfile_*` | 两份文件只差 `bazel build -s`（`-s` = `--subcommands`，仅打印编译命令） | 维护两份几乎相同的文件 | 确认 | 7.4 |
| 14 | `/root/brain`（构建上下文根） | 没有 `.dockerignore` | 本地临时文件、`__pycache__`、本地 node_modules/third_party 被拷进镜像，缓存易失效 | 确认 | 2.1 |
| 15 | `brain-service/scripts/dev.sh` | `--command bash -c "$CMD"` 中 `-c` 即 `--command`，bash 被覆盖 | 复合命令（`&&`、管道）无法执行 | 确认（argparse 实测） | 4.4 |
| 16 | `brain-service/scripts/dev.sh` | `docker exec -it name "$CMD"`；`--filter name=` 子串匹配 | 带参数的命令 exec 失败；名字前缀相同时误判 | 确认 | 4.5、5.9 |
| 17 | 各 build.sh | Harbor 用 `-p` 明文传密码；`docker -H tcp://` | 密码暴露在进程列表；非 TLS 的远程 daemon 端口 | 确认 | 7.2 |
| 18 | `robot-data-platform/Dockerfile` | 官方 python 镜像里又 apt 装 `python3-pip` | 多装一个用不到的 Python 解释器 | 确认 | 2.3 |
| 19 | `event-uploader/Dockerfile` | gcc 装了没卸；`From` 大小写；注释残留 `[cite: 176]` | 镜像偏大；代码质量 | 确认 | 2.3 |
| 20 | `video-monitoring/local-streaming-video/*` | 相机密码硬编码进 git | 凭据泄漏 | 确认 | 6.6 |
| 21 | `deploy.sh` vs `README.md` | 一个用 `docker-compose`（v1），一个用 `docker compose`（v2） | 新机器上 deploy.sh 直接退出 | 确认 | 6.1 |
| 22 | 多个 Dockerfile | `ENV KEY value` 旧语法；`FROM ... as` 大小写不一致 | BuildKit 检查警告 | 实测（`FromAsCasing`、`LegacyKeyValueFormat`） | 1.1、1.5 |

## 8.2 最佳实践清单

### Dockerfile

- [ ] 基础镜像写明 tag（最好钉 digest），用 `-slim` 等小镜像。
- [ ] 指令顺序：**稳定的在前、易变的在后**。依赖清单先 COPY、装依赖，再 COPY 源码。
- [ ] `apt-get update && apt-get install -y --no-install-recommends ... && rm -rf /var/lib/apt/lists/*` 写在一个 RUN 里。
- [ ] 下载 → 解压 → 删除包写在一个 RUN 里。
- [ ] 编译和运行分离：多阶段构建，最终阶段只 `COPY --from` 产物。
- [ ] 下载的东西钉版本号，不用 `latest` URL。
- [ ] 用 `COPY` 不用 `ADD`（除非要自动解压）。
- [ ] `ENV KEY=value`，`FROM ... AS name`，大小写一致。
- [ ] 版本信息用 `LABEL org.opencontainers.image.revision=...`，并真的从构建脚本传进来。
- [ ] 密钥用 `RUN --mount=type=secret`，不用 ARG/ENV。
- [ ] `CMD`/`ENTRYPOINT` 用 JSON 数组（exec 形式）。
- [ ] ENTRYPOINT 包装脚本最后一行 `exec "$@"` 或 `exec 程序`。
- [ ] 程序处理 SIGTERM，或者运行时加 `--init`。
- [ ] 构建期自测（`RUN app --help`），把问题挡在构建阶段。
- [ ] 有条件的话用非 root `USER`（机器人开发容器常常做不到，但生产服务值得考虑）。
- [ ] 一份 Dockerfile 支持多架构（`TARGETARCH`），不维护 `_x86_64` / `_aarch64` 两份。
- [ ] 每个构建上下文根目录都有 `.dockerignore`。
- [ ] 用 hadolint 检查：`docker run --rm -i hadolint/hadolint < Dockerfile`。
- [ ] 用 `docker build --check .` 检查（Docker 27+ / BuildKit 0.15+）。

### 构建脚本

- [ ] `set -euo pipefail`。
- [ ] 不认识的参数要报错（`build_docker.sh` 的 `*) ;;` 会静默忽略拼错的参数）。
- [ ] 临时文件用 `trap 'rm -rf ...' EXIT` 清理，避免中途失败后残留（rdp、event-uploader 的临时拷贝）。
- [ ] `docker login --password-stdin`。
- [ ] tag 命名约定只有一种。
- [ ] 推送多个仓库考虑仓库复制或 `skopeo copy`。

### 运行与编排

- [ ] 生产服务：`restart: unless-stopped`、`init: true`、日志轮转（`logging.options.max-size`）。
- [ ] 有依赖关系的服务配 `healthcheck` + `condition: service_healthy`。
- [ ] 端口只绑必要的地址（`127.0.0.1:port:port`）。
- [ ] 密码用 secrets 文件，不放环境变量、不进 git。
- [ ] 部署脚本不用 `down -v`。
- [ ] 部署时用明确的版本 tag，不用 `latest`。
- [ ] 开发容器才用 `--privileged`、`--network host`、挂 docker.sock；生产环境按需精确授权。

## 8.3 改写示例：brain-service

目标：**行为不变**（同样的基础镜像、同样的构建命令、同样的 ENTRYPOINT），修复 #1、#11、#12（部分）、#13、#14、#22。
这是一份参考写法，**没有在真实环境构建过**，替换前请先在宿主机上完整构建并跑一遍服务。

### `docker-deploy/brain-service/Dockerfile`（合并后的单个文件）

```dockerfile
# syntax=docker/dockerfile:1
ARG DEV_IMAGE=robot-harbor.geekplus.com/gino/brain-service-dev:v1.1.0
FROM ${DEV_IMAGE} AS brain-service

# 构建信息：由 build_docker.sh 传入（见下文脚本改动）
ARG GIT_COMMIT=unknown
ARG GIT_DIRTY_STATUS=unknown
ARG GIT_TIMESTAMP=0
ARG bazel_options=""

LABEL org.opencontainers.image.title="brain-service" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

ENV GIT_COMMIT=${GIT_COMMIT} \
    GIT_DIRTY_STATUS=${GIT_DIRTY_STATUS} \
    GIT_TIMESTAMP=${GIT_TIMESTAMP} \
    INSTANCE_NAME="brain_service"

WORKDIR /root/brain-service

# ① 与源码无关的大件放前面：只要基础镜像不变，这层永远命中缓存。
#    third_party / node_modules 在 git 里是被 ignore 的，源码里不会有同名内容，
#    所以先 cp 再 COPY 源码，结果与原来的顺序相同。
RUN cp -r /opt/geekplus/third_party /root/brain-service/ && \
    mkdir -p covariant/gui/ui && \
    cp -r /opt/geekplus/node_modules covariant/gui/ && \
    cp -r /opt/geekplus/ui_node_modules/node_modules covariant/gui/ui/

COPY fe_files /var/covariant/managed_data/fe_files/

# ② 源码：天天变，放在后面
COPY brain-service .

# ③ 编译。原 aarch64 版本多一个 -s（--subcommands，只是打印编译命令），
#    需要时用 BAZEL_OPTIONS="-s" 传入即可，不必为此维护两份 Dockerfile。
RUN bazel build ${bazel_options} //covariant/service:launch

RUN cd covariant/gui && yarn && yarn build

ENTRYPOINT ["/root/brain-service/docker-deploy/brain-service/scripts/run_brain.sh"]
```

### `docker-deploy/brain-service/Dockerfile.dockerignore`

BuildKit 支持 "`<Dockerfile 文件名>.dockerignore`" 放在 Dockerfile 旁边，**只对这个 Dockerfile 生效**（实测：`d/Build.df` 旁边的 `Build.df.dockerignore` 生效，上下文在别的目录也一样），
这样不需要在 `/root/brain` 根目录放文件。路径相对于**上下文根**（`/root/brain`）：

```
.git
bamboo-specs
source-deploy
docker
brain-service/bazel-*
brain-service/third_party
**/node_modules
**/__pycache__
**/*.pyc
**/.pytest_cache
**/.mypy_cache
```

`docker` 是 `--export` 输出 tar 包的目录（`$work_dir/docker/`），不排除的话下次构建会把上一次导出的几 GB tar 包拷进镜像。

### `run_brain.sh`

```bash
#!/bin/bash
set -e
python /opt/generate_ueye_conf.py
service ueyeethdrc start
exec python /root/brain-service/bazel-bin/covariant/service/launch "$@"
```

并且确认 launch 处理 SIGTERM（或者运行时加 `--init` / compose `init: true`）。

### `build_docker.sh` 的改动

```bash
# 选择 Dockerfile：不再区分架构
--file=./brain-service/docker-deploy/brain-service/Dockerfile

# 传入构建信息
build_args+=(--build-arg "GIT_COMMIT=$(git -C "$work_dir" rev-parse HEAD)")
build_args+=(--build-arg "GIT_TIMESTAMP=$(git -C "$work_dir" log -1 --format=%ct)")
if [[ -n "$(git -C "$work_dir" status --porcelain)" ]]; then
    build_args+=(--build-arg "GIT_DIRTY_STATUS=dirty")
else
    build_args+=(--build-arg "GIT_DIRTY_STATUS=clean")
fi

# 修复 --export
docker save "$IMAGE:${TAGS[0]}" -o "$exported_dir/brain-service.docker-image.tar"

# --dev：要么删除这个选项，要么改成直接 docker pull dev 镜像
```

### 进一步：bazel 缓存挂载（可选，需要验证）

如果构建机是固定的，可以让 bazel 缓存跨构建保留，改一行源码不必从头编译：

```dockerfile
RUN --mount=type=cache,target=/root/.cache/bazel,id=brain-bazel \
    bazel build ${bazel_options} //covariant/service:launch && \
    mkdir -p /opt/brain && \
    cp -rL bazel-bin/covariant/service/launch \
           bazel-bin/covariant/service/launch.runfiles /opt/brain/
```

代价（第 2、3 章讲过）：缓存目录不进镜像，`bazel-bin` 在镜像里会变成悬空链接，所以必须在同一个 RUN 里
`cp -rL` 把产物实体化拷出来，`run_brain.sh` 也要改成执行 `/opt/brain/launch`。
如果有其他代码依赖 `/root/brain-service/bazel-bin/...` 路径，也要一起改。
在 CI 上（agent 每次都是新容器、构建跑在宿主机 daemon 上）BuildKit 缓存是否能保留，取决于 daemon 是否长期运行，需要实际测试。

## 8.4 改写示例：几处小修复

**model-bin / infra-bin Dockerfile（#10）**：

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BAZEL_BIN=.                           # ← 移到 FROM 之后
WORKDIR /model-bin
COPY ${BAZEL_BIN} bazel-bin
WORKDIR /model-bin/bazel-bin/covariant/model
COPY entrypoint.sh entrypoint.sh
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
```

（顺便删掉了多余的 `RUN mkdir`：WORKDIR 会自动创建目录。）

**robot-data-platform（#18、#22）**：

```dockerfile
FROM robot-harbor.geekplus.com/gino/python:3.11-slim

ENV VIRTUAL_ENV=/env \
    PATH=/env/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY requirements/base.txt /app/base.txt
COPY requirements/local.txt /app/requirements.txt
RUN python -m venv /env && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY robot_data_platform robot_data_platform
COPY temp_rdp_shared_structs rdp_shared_structs
COPY config config
COPY manage.py .

EXPOSE 8000
CMD ["gunicorn", "--bind", ":8000", "--workers", "3", "config.wsgi:application"]
```

改成 `-slim` 前需确认依赖里没有需要编译的包（如 `psycopg2` 非 binary 版本需要 gcc 和 libpq-dev）；
如果有，用多阶段构建（第 3 章 3.2 末尾的 wheel 方案）。

**universal-uploader compose（#5～#8）**：

```yaml
name: universal-uploader

services:
  universal_uploader:
    container_name: universal-uploader
    restart: always
    init: true
    depends_on:
      db:
        condition: service_healthy
    environment:
      - MANAGED_DATA_S3_BUCKET=${BUCKET_PREFIX}-internal-data-us     # 去掉引号
      # ……其余不变
    command: |
      bash -c "
        export UNIVERSAL_UPLOADER_POSTGRES_USER=$$(cat /run/secrets/universal_uploader_postgres_user) &&
        export UNIVERSAL_UPLOADER_POSTGRES_PASSWORD=$$(cat /run/secrets/universal_uploader_postgres_password) &&
        exec /universal_uploader/launcher --config /var/covariant/universal_uploader/config.yaml
      "
    logging:
      options:
        max-size: "100m"
        max-file: "5"

  uploader_finalize_orphaned_files:
    restart: always
    init: true
    # ……

  db:
    ports:
      - "127.0.0.1:6432:6432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -p 6432"]
      interval: 5s
      timeout: 3s
      retries: 20
    # ……
```

## 8.5 学完之后可以做的事

1. 把 8.1 表里"确认"的问题挑几个提 PR（#1、#2、#4、#5 改动小、收益明确）。
3. 用 hadolint 扫一遍仓库所有 Dockerfile：
   ```bash
   cd /root/brain/brain-service
   find . -name 'Dockerfile*' -not -path './bazel-*' -not -path '*/node_modules/*' | \
     while read f; do echo "== $f"; docker run --rm -i hadolint/hadolint < "$f"; done
   ```
4. 用 `dive` 打开 brain-service 镜像，找出最大的几层，评估第 3 章 3.6 的多阶段改造是否值得做。

下一章：[09_cheatsheet.md](09_cheatsheet.md)
