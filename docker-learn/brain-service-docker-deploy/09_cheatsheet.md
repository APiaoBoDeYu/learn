# 第 9 章：命令速查

## 镜像

```bash
docker images                                  # 列出本地镜像
docker pull <img>:<tag>                        # 拉取
docker pull --platform linux/arm64 <img>       # 拉取指定架构
docker build -t name:tag -f Dockerfile <ctx>   # 构建
docker build --target <stage> ...              # 只构建到某个阶段
docker build --build-arg K=V ...               # 传构建参数
docker build --no-cache ...                    # 不用缓存
docker build --pull ...                        # 检查基础镜像是否有更新
docker build --progress=plain ...              # 展开日志（看 CACHED）
docker build --check .                         # 只做静态检查（Docker 27+）
docker tag src:tag registry/ns/name:tag        # 加一个名字
docker push registry/ns/name:tag               # 推送
docker history --no-trunc <img>                # 各层指令与大小
docker image inspect <img>                     # 元数据
docker image inspect -f '{{.Config.Entrypoint}} {{.Config.Cmd}}' <img>
docker image inspect -f '{{.Os}}/{{.Architecture}}' <img>
docker rmi <img>                               # 删除
docker image prune                             # 删除悬空镜像（<none>:<none>）
docker save img:tag -o img.tar                 # 导出（离线分发）
docker save img:tag | gzip > img.tar.gz
docker load -i img.tar                         # 导入
```

## 容器

```bash
docker run --rm -it <img> bash                 # 临时交互
docker run -d --name svc --restart unless-stopped <img>
docker run --rm -it --entrypoint bash <img>    # 绕过 ENTRYPOINT 进去排错
docker run --init ...                          # tini 做 PID 1
docker ps / docker ps -a
docker logs -f --tail 100 <c>
docker exec -it <c> bash
docker exec -it <c> bash -c "cmd1 && cmd2"
docker stop [-t 30] <c> / docker start <c> / docker restart <c>
docker rm [-f] <c>
docker inspect <c>
docker inspect -f '{{.State.ExitCode}} {{.State.OOMKilled}}' <c>
docker inspect -f '{{json .Mounts}}' <c>
docker top <c>
docker stats
docker cp <c>:/path ./local
docker diff <c>
docker container prune                         # 删除所有已停止容器
```

## `docker run` 常用参数

| 参数 | 含义 |
|---|---|
| `-p 8080:80` / `-p 127.0.0.1:8080:80` | 端口映射（宿主机:容器） |
| `--network host` | 宿主机网络 |
| `--network <net>` | 接入用户自定义网络（可用容器名互访） |
| `-v /host:/ctr[:ro]` | bind mount |
| `-v vol:/ctr` | named volume |
| `--mount type=bind,source=...,target=...` | 长写法；源不存在会报错 |
| `--tmpfs /tmp` | 内存文件系统 |
| `-e K=V` / `-e K` / `--env-file f` | 环境变量 |
| `-w /dir` | 工作目录 |
| `-u $(id -u):$(id -g)` | 运行用户 |
| `--gpus all` / `--runtime nvidia` | GPU |
| `--ipc host` / `--shm-size 8g` | 共享内存 |
| `--privileged` | 全部权限 |
| `--device /dev/xxx` | 单个设备 |
| `--cap-add NET_ADMIN` | 单项能力 |
| `--pid host` | 宿主机 PID 空间 |
| `--memory 8g --cpus 4` | 资源限制 |
| `--platform linux/amd64` | 以指定架构运行（需要 QEMU binfmt） |

## Compose（v2）

```bash
docker compose up -d [svc]
docker compose down                 # 不要随手加 -v
docker compose ps
docker compose logs -f [svc]
docker compose exec <svc> bash
docker compose run --rm <svc> <cmd> # 一次性任务
docker compose pull
docker compose config               # 查看替换后的最终配置
docker compose config --images      # 列出用到的镜像
docker compose -p <项目名> -f a.yml -f b.yml up -d
```

## 网络与卷

```bash
docker network ls
docker network create <net>
docker network inspect <net>
docker volume ls
docker volume inspect <vol>
docker volume rm <vol>
```

## 多架构 / 仓库

```bash
docker buildx ls
docker buildx create --use --name multi
docker buildx build --platform linux/amd64,linux/arm64 -t reg/img:tag --push .
docker buildx imagetools inspect reg/img:tag
docker manifest inspect reg/img:tag
docker manifest create reg/img:tag reg/img:tag_x86_64 reg/img:tag_aarch64
docker manifest push reg/img:tag
docker run --privileged --rm tonistiigi/binfmt --install all   # 安装 QEMU 模拟器
```

```bash
# 登录
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_DOCKER_REGISTRY
echo "$PASS" | docker login robot-harbor.geekplus.com -u "$USER" --password-stdin
```

## 清理磁盘

```bash
docker system df                    # 镜像/容器/卷/构建缓存各占多少
docker builder prune                # 清构建缓存
docker system prune                 # 停止的容器 + 悬空镜像 + 未用网络 + 构建缓存
docker system prune -a              # ……+ 所有没有容器在用的镜像（慎用）
```

## Dockerfile 模板（Python 服务）

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

FROM base AS production
COPY --from=builder /wheels /wheels
RUN pip install /wheels/* && rm -rf /wheels
COPY src/ src/
USER nobody
EXPOSE 8000
ENTRYPOINT ["python", "-m", "src.main"]
```

## 包装脚本模板

```bash
#!/bin/bash
set -euo pipefail
# 启动前准备工作……
exec "$@"
```
