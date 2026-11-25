# 从提交代码到机器人上跑起来：VLA 镜像的构建与部署流程

本教程讲清楚 `embodied_algo` 仓库里三类文件之间的关系：

| 文件 | 管什么 |
|---|---|
| `bamboo-specs/bamboo.yaml` | CI：代码推送后自动做什么 |
| `docker-deploy/vla/` | 镜像怎么构建（脚本和 Dockerfile） |
| `body/gino/docker-compose/docker-compose.yml` | 容器在机器人上怎么启动 |

读完后你应该能回答：推送一个提交后，会不会构建镜像？镜像 tag 是什么？机器人上怎么用到这个镜像？

## 一句话总结

三者之间**没有直接调用**。CI 构建镜像并推到镜像仓库，docker-compose 从镜像仓库拉镜像启动容器，两边只靠**镜像名加 tag** 对上号。

## 目录

| 章节 | 内容 |
|---|---|
| [01-CI-CD概念.md](01-CI-CD概念.md) | CI、持续交付、持续部署分别是什么，这个项目做到哪一步 |
| [02-bamboo流水线.md](02-bamboo流水线.md) | `bamboo.yaml` 的结构和三个阶段，script 任务怎么执行，镜像构建脚本逐段讲解 |
| [03-构建脚本.md](03-构建脚本.md) | `should_build.sh` 怎么决定构建什么，`build.sh` 怎么构建和推送 |
| [04-docker-compose启动.md](04-docker-compose启动.md) | compose 是什么，逐项讲解（body 为什么只启动 bash、stdin_open/tty、ulimits、logging、bind 挂载），以及它和 Dockerfile 怎么配合 |
| [05-端到端流程.md](05-端到端流程.md) | 把前面串起来：一次发版从头到尾发生了什么，附自测题 |

## 相关教程

- Dockerfile.release 的语法和设计：`/root/workspace/docker-learn/vla-dockerfile-release/`
- 上一级目录是以 `brain-service/docker-deploy` 为教材的 Docker 系统教程，其中第 06 章讲 Compose、第 07 章讲 tag 和镜像仓库，可以和本教程对照着看。

本教程的内容基于阅读配置文件和脚本整理，流程没有在 Bamboo 和机器人上实际跑过。
