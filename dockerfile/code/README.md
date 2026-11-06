# 动手实验

所有实验都需要在**装了 Docker 的宿主机**上运行（Docker 23+，默认启用 BuildKit；lab07 需要 buildx）。

全部实验已在 Jetson Thor（aarch64，Docker 29.1.4 / BuildKit v0.26.3 / buildx v0.30.1 / Compose v5.0.1）上跑通，
各 README 中"实测"部分的数字来自这台机器。换机器后绝对耗时会变，结论应该不变。

| 实验 | 对应章节 | 内容 | 耗时 |
|---|---|---|---|
| [lab01_basics](lab01_basics/) | 第 1 章 | 最小 Python 服务镜像；观察层与元数据 | 10 分钟 |
| [lab02_cache](lab02_cache/) | 第 2 章 | 指令顺序对缓存的影响；"删了但没变小" | 15 分钟 |
| [lab03_multistage](lab03_multistage/) | 第 3 章 | 单阶段 vs 多阶段体积对比；`--target` | 10 分钟 |
| [lab04_signals](lab04_signals/) | 第 4 章 | 五种 ENTRYPOINT 写法下 `docker stop` 的耗时 | 10 分钟 |
| [lab05_arg_scope](lab05_arg_scope/) | 第 1 章 1.6 | 验证 infra-bin/model-bin 的 ARG 作用域问题 | 5 分钟 |
| [lab06_compose](lab06_compose/) | 第 6 章 | 应用 + Postgres + secrets + 健康检查；引号陷阱；`down -v` | 20 分钟 |
| [lab07_multiarch](lab07_multiarch/) | 第 7 章 | buildx 双架构构建 + 本地仓库 + manifest list | 20 分钟 |

## 关于基础镜像

实验默认使用 Docker Hub 上的官方镜像（`python:3.11-slim`、`debian:bookworm-slim`、`postgres:16`、`registry:2`）。
如果拉不下来，可以换成公司 Harbor 上的同步镜像，例如：

```bash
docker pull robot-harbor.geekplus.com/gino/python:3.11-slim
docker tag  robot-harbor.geekplus.com/gino/python:3.11-slim python:3.11-slim
```

Dockerfile 里的基础镜像大多做成了 `ARG BASE=...`，也可以直接 `--build-arg BASE=robot-harbor.geekplus.com/gino/python:3.11-slim`。

## 清理

```bash
docker rm -f $(docker ps -aq --filter "name=lab") 2>/dev/null
docker images | grep '^lab' | awk '{print $1":"$2}' | xargs -r docker rmi
docker builder prune -f
```
