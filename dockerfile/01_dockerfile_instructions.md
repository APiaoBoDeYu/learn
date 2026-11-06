# 第 1 章：Dockerfile 指令逐条精讲

Dockerfile 一共不到 20 条指令，常用的只有 10 条。本章按"出现频率"排序，每条都用仓库里的真实代码讲。

先把仓库里三个风格不同的 Dockerfile 摆出来，后文反复引用：

**A. `robot-data-platform/Dockerfile`**（最"教科书"，Django + gunicorn）

```dockerfile
FROM robot-harbor.geekplus.com/gino/python:3.11

COPY requirements/base.txt /app/base.txt
COPY requirements/local.txt /app/requirements.txt

RUN set -ex \
    && apt-get update \
    && apt-get install -y python3-pip --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /env \
    && /env/bin/pip install --upgrade pip \
    && /env/bin/pip install --no-cache-dir -r /app/requirements.txt

COPY robot_data_platform /app/robot_data_platform
COPY temp_rdp_shared_structs /app/rdp_shared_structs
COPY config /app/config
COPY manage.py /app

WORKDIR /app

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ENV PYTHONUNBUFFERED 1

EXPOSE 8000
CMD ["gunicorn", "--bind", ":8000", "--workers", "3", "config.wsgi:application"]
```

**B. `brain-service/Dockerfile_x86_64`**（在镜像里编译）

```dockerfile
FROM robot-harbor.geekplus.com/gino/brain-service-dev:v1.1.0 as brain-service
WORKDIR /root/brain-service

ARG GIT_COMMIT
ENV GIT_COMMIT ${GIT_COMMIT}
ARG GIT_DIRTY_STATUS
ENV GIT_DIRTY_STATUS ${GIT_DIRTY_STATUS}
ARG GIT_TIMESTAMP
ENV GIT_TIMESTAMP ${GIT_TIMESTAMP:-0}
ENV INSTANCE_NAME="brain_service"

ARG bazel_options=""

ENTRYPOINT [ "/root/brain-service/docker-deploy/brain-service/scripts/run_brain.sh" ]

COPY brain-service .
COPY fe_files /var/covariant/managed_data/fe_files/
RUN cp -r /opt/geekplus/third_party /root/brain-service/
RUN cp -r /opt/geekplus/node_modules /root/brain-service/covariant/gui/
RUN cp -r /opt/geekplus/ui_node_modules/node_modules /root/brain-service/covariant/gui/ui/

RUN bazel build \
    $bazel_options \
    //covariant/service:launch

RUN cd covariant/gui && yarn && yarn build && cd -
```

**C. `universal-uploader/Dockerfile_x86_64`**（装 conda，只拷产物，带构建期自测）—— 节选

```dockerfile
FROM robot-harbor.geekplus.com/gino/ubuntu:22.04
ARG BAZEL_BIN
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && ...
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl ... \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/conda/envs/lab42/bin:/opt/conda/bin:${PATH}
RUN curl -fsSL -o ~/miniforge.sh https://github.com/.../Miniforge3-Linux-x86_64.sh && ...
COPY ${BAZEL_BIN}/requirements.txt /requirements.txt
RUN conda run -n lab42 uv pip install psycopg2-binary -r /requirements.txt
COPY ${BAZEL_BIN}/covariant/infra/universal_uploader /universal_uploader
ENV RUNFILES_DIR /universal_uploader/launcher.runfiles
RUN DISABLE_LOGGING=1 RDP_URL=DUMMY /universal_uploader/launcher --help
CMD ["/universal_uploader/launcher"]
```

---

## 1.1 `FROM` —— 从哪个镜像开始

```dockerfile
FROM <镜像>[:<tag>|@<digest>] [AS <阶段名>]
```

- 每个 Dockerfile 必须以 `FROM` 开始（前面只能有 `ARG` 和注释/解析指令）。
- **镜像名的完整结构**：`[registry/][namespace/]repository[:tag]`

  ```
  robot-harbor.geekplus.com / gino / brain-service-dev : v1.1.0
  └──── registry ─────────┘ └ ns ┘ └── repository ──┘   └ tag ┘
  ```

  省略 registry 时默认是 Docker Hub（`docker.io/library/`）。仓库里所有基础镜像都写成
  `robot-harbor.geekplus.com/gino/python:3.11` 这样，是因为公司把 Docker Hub 上的官方镜像
  **同步到了内部 Harbor**——国内直连 Docker Hub 不稳定，也便于管控版本。

- **tag 是可变的，digest 是不可变的。** `python:3.11` 今天和三个月后可能指向不同内容；
  `python@sha256:abc...` 永远是同一份。生产环境追求可复现时会钉 digest。

- `AS brain-service` 给这个阶段起名，用于多阶段构建和 `docker build --target`（第 3 章）。
  仓库里写成了小写 `as`，能用，但新版 BuildKit 会提示 `FromAsCasing` 警告（`FROM` 大写时 `AS` 也要大写）。
  `event-uploader/Dockerfile` 里写的是 `From`，同理：**指令不区分大小写，但约定全大写**。

- `FROM` 可以用变量（见 1.6 节 ARG）：`model-bin/Dockerfile` 的 `FROM ${BASE_IMAGE}`
  让同一个 Dockerfile 能基于不同的基础镜像构建，`build.sh --base-img xxx` 传入。

## 1.2 `RUN` —— 构建时执行命令，产生一层

```dockerfile
RUN <命令>                       # shell 形式：实际执行 /bin/sh -c "<命令>"
RUN ["可执行文件", "参数1", ...]   # exec 形式：不经过 shell，没有变量展开/管道/&&
```

绝大多数 `RUN` 用 shell 形式。几个必须知道的细节：

**① 每个 `RUN` 是一个全新的 shell，互不继承状态。**

```dockerfile
RUN cd covariant/gui && yarn && yarn build && cd -
```

这里末尾的 `cd -` 没有意义：这个 RUN 结束 shell 就退出了，下一个 RUN 照样从 `WORKDIR` 开始。
同理，`RUN export FOO=1` 对后续 RUN 无效 —— 要跨指令持久的环境变量必须用 `ENV`。
要改变后续指令的工作目录必须用 `WORKDIR`。

**② 默认 shell 是 `/bin/sh`（Ubuntu/Debian 上是 dash），不是 bash。**
`[[ ]]`、`source`、数组、`{a,b}` 花括号展开在 dash 里都不能用。需要 bash 时可以
`SHELL ["/bin/bash", "-c"]` 或 `RUN bash -c '...'`。

**③ 一个 RUN 里任意一步失败，就要让整条失败。**
`A && B && C` 是最常见写法。A 那个 Dockerfile 用的 `set -ex` 是另一种思路：
`-e` 遇错即停，`-x` 把每条命令打印到构建日志，排查问题非常方便。
小心管道：`curl ... | tar x` 中 `curl` 失败了整条命令的退出码仍是 `tar` 的，
需要 `set -o pipefail`（dash 不支持，要换 bash）。

**④ "装完就清"必须在同一个 RUN 里。**

```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl ... \
    && rm -rf /var/lib/apt/lists/*
```

这是 C 文件的写法，正确。如果把 `rm -rf /var/lib/apt/lists/*` 写到下一个 RUN，
apt 的索引（几十 MB）已经固化在上一层里了，删了也不会让镜像变小（第 0 章"只增不减"）。

`--no-install-recommends` 让 apt 不装"推荐"的附带包，通常能省不少体积。

**⑤ `apt-get update` 和 `apt-get install` 也必须在同一个 RUN 里。**
如果分开，`update` 那层会被缓存，几个月后再改 `install` 那行时，用的还是旧索引，
可能 404。第 2 章讲缓存时会再说。

## 1.3 `COPY` / `ADD` —— 把构建上下文里的文件放进镜像

```dockerfile
COPY <源...> <目标>
COPY --chown=user:group <源> <目标>
COPY --from=<阶段名或镜像> <源> <目标>      # 多阶段构建，第 3 章
```

- **源路径相对于构建上下文**（不是相对于 Dockerfile 所在目录！）。这一点在第 2 章详细展开。
- **目标路径**：绝对路径，或相对于当前 `WORKDIR`。
  B 文件里 `WORKDIR /root/brain-service` 之后的 `COPY brain-service .`，目标就是 `/root/brain-service`。
- **目录的复制规则**：`COPY dir /x` 复制的是 dir **里面的内容**到 /x，不会生成 `/x/dir`。
  所以 `COPY robot_data_platform /app/robot_data_platform` 要把目录名在目标里重写一遍。
- **目标以 `/` 结尾表示目录**：`COPY fe_files /var/covariant/managed_data/fe_files/`。
  多个源文件时目标必须是目录且以 `/` 结尾。
- `COPY` 会自动创建不存在的目标目录，所以 `model-bin/Dockerfile` 里的 `RUN mkdir /model-bin` 是多余的
  （`WORKDIR` 也会自动创建目录）。
- 复制进来的文件默认属主是 root。容器以非 root 用户运行时用 `--chown`。

**`ADD` 和 `COPY` 的区别**：`ADD` 额外能自动解压本地 tar 包、能从 URL 下载。
正是因为这些"魔法"行为不直观，**官方建议除非需要解压 tar，否则一律用 COPY**。
仓库里全部用的是 COPY，正确。

## 1.4 `WORKDIR` —— 设置后续指令和容器启动时的工作目录

```dockerfile
WORKDIR /app
```

- 影响其后的 `RUN`、`CMD`、`ENTRYPOINT`、`COPY`（相对目标路径），以及**容器启动时的默认目录**。
- 不存在会自动创建。
- 可以写多次，相对路径基于上一个 WORKDIR。

`model-bin/Dockerfile` 最后 `WORKDIR /model-bin/bazel-bin/covariant/model`，配合
`ENTRYPOINT ["/bin/bash", "entrypoint.sh"]`（相对路径），意思是"在这个目录下找 entrypoint.sh"。
这样用户 `docker run model-bin scripts/dataset/_export_v2_dataset ...` 时也能用相对路径调用 bazel 产物。

**不要用 `RUN cd xxx` 代替 WORKDIR**（原因见 1.2 ①）。

## 1.5 `ENV` —— 设置环境变量（构建期和运行期都生效）

```dockerfile
ENV KEY=value                 # 推荐写法
ENV KEY1=v1 KEY2="v 2"        # 一次多个
ENV KEY value                 # 旧写法，仍可用，但 BuildKit 会给 LegacyKeyValueFormat 警告
```

仓库里 A、B、C 三个文件都混用了新旧写法（`ENV PATH /env/bin:$PATH`、`ENV INSTANCE_NAME="brain_service"`）。
（实测 `docker build --check` 对 `ENV A b` 报 `LegacyKeyValueFormat`，对 `FROM busybox as base` 报 `FromAsCasing`。）
旧写法的问题是：`ENV A B C` 到底是 A="B C" 还是什么，读者要想一下。统一用 `=`。

ENV 的值会：
1. 对后续所有构建指令可见（`RUN` 里能用）；
2. **写进镜像元数据**，容器运行时默认就有；
3. 可以被 `docker run -e KEY=xxx` 覆盖。

A 文件里三条 ENV 都很典型：

| ENV | 作用 |
|---|---|
| `PATH /env/bin:$PATH` | 把 venv 放到 PATH 最前面，之后 `gunicorn`、`python` 都指向 venv 里的。效果等同于"永久 activate" |
| `VIRTUAL_ENV /env` | 一些工具靠它判断当前是否在 venv 里 |
| `PYTHONUNBUFFERED 1` | 让 Python 不缓冲 stdout。**容器里跑 Python 几乎必加**，否则 `docker logs` 看不到实时输出（compose 文件里也专门注释了这件事） |

C 文件里有一段注释很值得读：

> Make sure /opt/conda/envs/lab42/bin is BEFORE /opt/conda/bin, as both contain a python binary

用 `ENV PATH=...` 控制可执行文件的查找顺序，是容器里"激活环境"的标准方法。
`echo "conda activate lab42" >> ~/.bashrc` 只对交互式 bash 生效，`CMD` 启动的进程不读 `.bashrc`，
所以真正起作用的是那条 ENV PATH。

## 1.6 `ARG` —— 构建参数（只在构建期存在）

```dockerfile
ARG NAME                # 没有默认值
ARG NAME=default        # 有默认值
```

```bash
docker build --build-arg NAME=xxx .
```

### ARG vs ENV

| | `ARG` | `ENV` |
|---|---|---|
| 构建期 RUN 中可用 | ✅ | ✅ |
| 进入镜像元数据、容器运行时可见 | ❌ | ✅ |
| 外部传入方式 | `docker build --build-arg` | `docker run -e`（运行时） |

B 文件演示了一个常见组合：**用 ARG 接收、用 ENV 固化**。

```dockerfile
ARG GIT_COMMIT
ENV GIT_COMMIT ${GIT_COMMIT}
ARG GIT_TIMESTAMP
ENV GIT_TIMESTAMP ${GIT_TIMESTAMP:-0}     # 支持 ${VAR:-default} 语法
```

意图是把构建时的 git commit 写进镜像，运行时服务能报告自己是哪个版本。
**但 `build_docker.sh` 从来没有传 `--build-arg GIT_COMMIT=...`**（它只传了 HTTP_PROXY、HTTPS_PROXY、bazel_options），
所以实际构建出的镜像里 `GIT_COMMIT` 是空串、`GIT_TIMESTAMP` 是 0。要让它生效，脚本里需要加：

```bash
build_args+=(--build-arg "GIT_COMMIT=$(git rev-parse HEAD)")
build_args+=(--build-arg "GIT_TIMESTAMP=$(git log -1 --format=%ct)")
```

更标准的做法是写进 `LABEL`（见 1.9）。

### 预定义的代理 ARG

各个 build.sh 都有这段：

```bash
if [[ -n "$HTTP_PROXY" ]]; then
    build_args+=(--build-arg "HTTP_PROXY=$HTTP_PROXY")
fi
```

但 Dockerfile 里并没有声明 `ARG HTTP_PROXY` —— 这能生效吗？能。`HTTP_PROXY`、`HTTPS_PROXY`、
`NO_PROXY`、`FTP_PROXY`（及小写形式）是 Docker **预定义的 ARG**，不需要声明就能在 RUN 里使用，
而且**故意不记录进 `docker history`**，避免代理地址（可能含密码）泄漏到镜像里。

### ⚠ ARG 的作用域（仓库里的一个隐患）

规则：**`FROM` 之前声明的 ARG 只能用在 `FROM` 行里**。想在 `FROM` 之后用，必须在阶段内再声明一次（不带值）。

```dockerfile
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}       # ✅ 可以用
RUN echo ${BASE_IMAGE}   # ❌ 空的
ARG BASE_IMAGE           # 重新"引入"进本阶段
RUN echo ${BASE_IMAGE}   # ✅ ubuntu:22.04
```

现在看 `model-bin/Dockerfile` 和 `infra-bin/Dockerfile`：

```dockerfile
ARG BASE_IMAGE
ARG BAZEL_BIN            # ← 在 FROM 之前声明
FROM ${BASE_IMAGE}
...
COPY ${BAZEL_BIN} bazel-bin    # ← FROM 之后使用，没有重新声明
```

**已实测（lab05，Docker 29.1.4）**：阶段内 `${BAZEL_BIN}` 是**空的**，`--build-arg BAZEL_BIN=.` 实际没起作用：

```
#6 0.516 inside stage: BASE_IMAGE='' BAZEL_BIN=''
```

这条 COPY 之所以"能用"，是因为空的源路径被当成了上下文根目录，复制结果和传入 `.` 完全一样 —— 碰巧能用。
把仓库里 `infra-bin/Dockerfile` 原文交给 `docker build --check`，会在 COPY 行报：

```
WARNING: UndefinedVar - Usage of undefined variable '$BAZEL_BIN'
```

`universal-uploader` 的 Dockerfile 把 `ARG BAZEL_BIN` 写在 `FROM` 之后，是正确写法。
修复方式：在 model-bin/infra-bin 的 `FROM` 后补一行 `ARG BAZEL_BIN`。

### 不要用 ARG 传密码

ARG 的值会出现在 `docker history` 里（预定义代理 ARG 除外）。传 token、密钥应该用
BuildKit 的 `RUN --mount=type=secret`（第 3 章）。

## 1.7 `CMD` / `ENTRYPOINT` —— 容器启动时跑什么

这两个是第 4 章的主题，这里只给结论：

```dockerfile
ENTRYPOINT ["可执行文件", "固定参数"]   # 容器"是什么程序"
CMD ["默认参数"]                       # 默认参数，docker run 后面跟的东西会替换它
```

容器启动的实际命令 = `ENTRYPOINT + CMD`。

- A 文件只有 `CMD`：`docker run rdp` 跑 gunicorn；`docker run rdp bash` 就进 shell（整个 CMD 被替换）。
- B 文件只有 `ENTRYPOINT`：`docker run brain-service launch-all` 等价于 `run_brain.sh launch-all`。
- **一律用 JSON 数组（exec 形式）**，原因在第 4 章。

B 文件把 `ENTRYPOINT` 写在了 `COPY` 之前。ENTRYPOINT 只修改元数据，不影响镜像内容，
但它**会参与缓存计算**：哪天改了 ENTRYPOINT 的路径，它后面的所有层（包括最慢的 bazel build）都要重建。
所以**惯例是放在文件末尾**：既不拖累缓存，读者也能一眼看到"这个镜像最终跑什么"。

## 1.8 `EXPOSE` —— 只是文档

```dockerfile
EXPOSE 8000
```

**EXPOSE 不会打开或映射任何端口**，只是在镜像元数据里声明"这个服务监听 8000"。
真正让外部访问需要 `docker run -p 8000:8000`，或者 `--network host`（第 5 章）。
`docker run -P`（大写）会把所有 EXPOSE 的端口随机映射出来，这是 EXPOSE 唯一的实际作用。

## 1.9 其他常用指令

| 指令 | 作用 | 仓库里 |
|---|---|---|
| `LABEL k=v` | 给镜像打元数据标签，`docker inspect` 可见。适合放 git commit、构建时间、维护者 | 没用到，可以替代 B 文件的 GIT_* ENV |
| `USER name` | 之后的 RUN/CMD/ENTRYPOINT 以该用户运行 | 没用到，全部以 root 运行（第 8 章讨论） |
| `VOLUME /path` | 声明挂载点；运行时没 `-v` 也会自动创建匿名卷 | `covariant/gui/deploy/Dockerfile` 的 `VOLUME /docker-console` |
| `HEALTHCHECK CMD ...` | 定义健康检查，`docker ps` 会显示 healthy/unhealthy；compose 的 `depends_on: condition: service_healthy` 依赖它 | 没用到（第 6 章建议加） |
| `SHELL ["/bin/bash", "-c"]` | 改变 shell 形式 RUN 使用的 shell | 没用到 |
| `STOPSIGNAL SIGINT` | `docker stop` 发送的信号（默认 SIGTERM） | 没用到 |
| `ONBUILD` | 给"以本镜像为基础"的下游构建埋指令 | 少见，不推荐 |

## 1.10 注释与解析指令

```dockerfile
# syntax=docker/dockerfile:1.7-labs     ← 解析指令，必须在文件第一行，指定 Dockerfile 前端版本
# hadolint ignore=DL3006                 ← 给 hadolint（Dockerfile 静态检查工具）的忽略标记
# 普通注释
```

`model-bin/Dockerfile` 里的 `# hadolint ignore=DL3006` 说明有人用 hadolint 检查过这个文件。
DL3006 是"FROM 的镜像应该写明 tag"——这里镜像名来自变量，hadolint 无法判断，所以忽略。
推荐你也装一个 hadolint，第 8 章用它把仓库的 Dockerfile 都扫一遍。

`local_clipping/Dockerfile` 第一行 `# syntax = robot-harbor.geekplus.com/gino/dockerfile:1.7-labs`
指定了 BuildKit 的 Dockerfile 前端版本（同步到了内部 Harbor），这样才能用 `COPY --exclude` 这种实验语法（第 3 章）。

## 1.11 一个小细节：C 文件的"构建期自测"

```dockerfile
ENV RUNFILES_DIR /universal_uploader/launcher.runfiles
RUN DISABLE_LOGGING=1 RDP_URL=DUMMY /universal_uploader/launcher --help
```

在构建的最后跑一次 `--help`：如果依赖缺失、路径不对、动态库加载失败，**构建直接失败**，
而不是等部署到工作站上才发现。代价只是几秒钟。非常值得学习的做法。

（`VAR=x cmd` 是 shell 语法：只对这一条命令设置环境变量，不影响镜像。）

## 1.12 动手

做 [code/lab01_basics](code/lab01_basics/)：写一个最小的 Python 服务镜像，用 `docker history`
和 `docker inspect` 观察每条指令产生了什么。
做 [code/lab05_arg_scope](code/lab05_arg_scope/)：亲手验证 ARG 作用域。

下一章：[02_context_and_cache.md](02_context_and_cache.md)
