# Dockerfile 由浅入深：以 VLA 发布镜像为例

本教程以 `embodied_algo/docker-deploy/vla/Dockerfile.release` 为最终目标，从零讲起，一步步补齐读懂它所需的知识。前几章每章配一个可以自己动手跑的小例子，最后一章逐行精读原文件。

## 目录

| 章节 | 内容 | 对应原文件的哪部分 |
|---|---|---|
| [01-基础概念.md](01-基础概念.md) | 镜像、容器、层、构建上下文 | 全文的前提 |
| [02-基础指令.md](02-基础指令.md) | `FROM` `RUN` `COPY` `WORKDIR` | builder 阶段的主体 |
| [03-变量与元数据.md](03-变量与元数据.md) | `ARG` `ENV` `LABEL` | 第 25-26、67-73 行 |
| [04-启动命令.md](04-启动命令.md) | `ENTRYPOINT` `CMD` | 第 89-90、117-118 行 |
| [05-构建上下文与dockerignore.md](05-构建上下文与dockerignore.md) | 上下文、`.dockerignore`、白名单写法 | `Dockerfile.release.dockerignore` |
| [06-多阶段构建.md](06-多阶段构建.md) | `FROM ... AS`、`COPY --from`、`--target` | 整体结构 |
| [07-构建期检查.md](07-构建期检查.md) | `set -eux`、`test`、冒烟测试 | 第 40-60、84-87 行 |
| [08-缓存与层.md](08-缓存与层.md) | 缓存规则、为什么分 env 和 release 两层镜像 | 整体设计 |
| [09-逐行精读.md](09-逐行精读.md) | 把前面的知识套到原文件上 | 全文 |

## 准备

- 安装 Docker，确认 `docker version` 能输出 Client 和 Server 两部分。
- Docker 23.0 及以上默认使用 BuildKit。版本更老的话，执行命令前先 `export DOCKER_BUILDKIT=1`。
- 练习用的镜像都基于 `python:3.12-slim`，在 x86_64 和 arm64 上都能跑。原文件的基础镜像只有 arm64 版本，只能在 Jetson Thor 上构建。

建议每章新建一个空目录做练习，例如 `mkdir -p ~/docker-lab/ch02 && cd ~/docker-lab/ch02`。
