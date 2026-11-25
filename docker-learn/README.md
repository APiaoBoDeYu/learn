# Docker 学习

两套教程，各自以一份真实在用的部署代码当教材：

| 目录 | 教材 | 内容 | 篇幅 |
|---|---|---|---|
| [vla-dockerfile-release/](vla-dockerfile-release/) | `embodied_algo/docker-deploy/vla/Dockerfile.release` | 只讲 Dockerfile，从基础指令讲到逐行精读一个文件 | 9 章，每章带小练习 |
| [brain-service-docker-deploy/](brain-service-docker-deploy/) | `brain-service/docker-deploy` 整套部署代码 | Dockerfile、`docker run`、Compose、推送与多架构、问题清单 | 10 章 + 7 个实测过的 lab |

## 阅读顺序

1. **先读 VLA 篇。** 篇幅短，只围绕一个 Dockerfile，适合第一次写 Dockerfile 的读者。读完能看懂一个多阶段的发布镜像。
2. **再读 brain-service 篇。** 前 4 章和 VLA 篇主题重叠，但讲得更深（PID 1 与信号、`RUN --mount`、ARG 作用域的实测复现），可以挑着读。第 5 章以后是 VLA 篇没有的内容：`docker run` 参数、Compose 编排、tag/推送/离线分发/多架构。

## 主题对照

同一主题在两套教程里的位置。章节文件里也加了"另见"链接，互相指向。

| 主题 | VLA 篇 | brain-service 篇 | 实验 |
|---|---|---|---|
| 镜像、容器、层 | [01-基础概念](vla-dockerfile-release/01-基础概念.md) | [00 心智模型](brain-service-docker-deploy/00_overview.md) | — |
| `FROM` `RUN` `COPY` `WORKDIR` | [02-基础指令](vla-dockerfile-release/02-基础指令.md) | [01 指令精讲](brain-service-docker-deploy/01_dockerfile_instructions.md) 1.1–1.4 | [lab01](brain-service-docker-deploy/code/lab01_basics/) |
| `ARG` `ENV` `LABEL`、ARG 作用域 | [03-变量与元数据](vla-dockerfile-release/03-变量与元数据.md) | [01 指令精讲](brain-service-docker-deploy/01_dockerfile_instructions.md) 1.5–1.6 | [lab05](brain-service-docker-deploy/code/lab05_arg_scope/) |
| `ENTRYPOINT` `CMD`、信号 | [04-启动命令](vla-dockerfile-release/04-启动命令.md) | [04 ENTRYPOINT 与信号](brain-service-docker-deploy/04_entrypoint_cmd_signals.md) | [lab04](brain-service-docker-deploy/code/lab04_signals/) |
| 构建上下文、`.dockerignore` | [05-构建上下文与dockerignore](vla-dockerfile-release/05-构建上下文与dockerignore.md) | [02 上下文与缓存](brain-service-docker-deploy/02_context_and_cache.md) 2.1 | — |
| 多阶段构建、`--target` | [06-多阶段构建](vla-dockerfile-release/06-多阶段构建.md) | [03 多阶段构建](brain-service-docker-deploy/03_multistage_and_targets.md) | [lab03](brain-service-docker-deploy/code/lab03_multistage/) |
| 构建期检查 | [07-构建期检查](vla-dockerfile-release/07-构建期检查.md) | [01 指令精讲](brain-service-docker-deploy/01_dockerfile_instructions.md) 1.11 | — |
| 层缓存、镜像体积 | [08-缓存与层](vla-dockerfile-release/08-缓存与层.md) | [02 上下文与缓存](brain-service-docker-deploy/02_context_and_cache.md) 2.2–2.3 | [lab02](brain-service-docker-deploy/code/lab02_cache/) |
| 逐行精读一个 Dockerfile | [09-逐行精读](vla-dockerfile-release/09-逐行精读.md) | [08 问题清单与改写](brain-service-docker-deploy/08_review_and_best_practices.md) | — |
| `docker run` 参数 | — | [05 docker run](brain-service-docker-deploy/05_docker_run_in_depth.md) | — |
| Docker Compose | — | [06 Compose](brain-service-docker-deploy/06_compose.md) | [lab06](brain-service-docker-deploy/code/lab06_compose/) |
| tag、推送、多架构 | — | [07 tag/push/多架构](brain-service-docker-deploy/07_tag_push_multiarch.md) | [lab07](brain-service-docker-deploy/code/lab07_multiarch/) |
| 命令速查 | — | [09 速查](brain-service-docker-deploy/09_cheatsheet.md) | — |
