# lab05：ARG 作用域（验证 infra-bin / model-bin 的写法）

对应第 1 章 1.6 节、第 8 章问题 #10。

`infra-bin/Dockerfile` 和 `model-bin/Dockerfile` 把 `ARG BAZEL_BIN` 写在 `FROM` 之前，
却在 `FROM` 之后的 `COPY ${BAZEL_BIN} bazel-bin` 里使用。按官方文档，FROM 之前的 ARG 只在 FROM 行里有效。
这个实验验证：(1) 阶段内 `${BAZEL_BIN}` 是否为空；(2) 为空时 `COPY  bazel-bin` 实际复制了什么。

```
bazel-bin-demo/           ← 模拟 infra-bin/build.sh 里 cd 进去的 bazel-bin 目录（构建上下文）
├── entrypoint.sh
└── covariant/infra/tool.txt
```

## 步骤

```bash
cd lab05_arg_scope

# 1. 仓库原写法，传入和 build.sh 一样的参数
docker build --no-cache --progress=plain \
    --build-arg BAZEL_BIN=. \
    -f Dockerfile.repo_style bazel-bin-demo 2>&1 | grep -E "inside stage|/infra-service|ERROR|warn|WARN"

# 2. 修复后的写法
docker build --no-cache --progress=plain \
    --build-arg BAZEL_BIN=. \
    -f Dockerfile.fixed bazel-bin-demo 2>&1 | grep -E "inside stage|/infra-service|ERROR"

# 3. 让 BuildKit 的静态检查说话（Docker 27+）
docker build --check --build-arg BAZEL_BIN=. -f Dockerfile.repo_style bazel-bin-demo
```

## 实测结果（Docker 29.1.4 / BuildKit v0.26.3）

| 问题 | 原写法（= infra-bin / model-bin） | 修复写法 |
|---|---|---|
| 阶段内 `BAZEL_BIN` 的值 | **`''`（空）** | `'.'` |
| COPY 是否成功 | 成功 | 成功 |
| `/infra-service/bazel-bin` 下的文件 | `covariant/infra/tool.txt`、`entrypoint.sh` | 同左 |
| `--check` | **`UndefinedVar: Usage of undefined variable '$BAZEL_BIN'`**（指向 COPY 行） | `no warnings found` |

原写法的构建日志：

```
#6 0.516 inside stage: BASE_IMAGE='' BAZEL_BIN=''
#9 0.910 /infra-service/bazel-bin/covariant/infra/tool.txt
#9 0.910 /infra-service/bazel-bin/entrypoint.sh
 1 warning found (use docker --debug to expand)
```

结论：

1. 文档规则得到证实：`FROM` 之前声明的 ARG 在阶段内是空的，`--build-arg BAZEL_BIN=.` **没有生效**。
2. `COPY  bazel-bin`（源路径为空）被当作**上下文根目录**，结果和 `COPY . bazel-bin` 完全一样 ——
   所以仓库里的 infra-bin / model-bin 目前"碰巧能用"。哪天有人想用 `--build-arg BAZEL_BIN=some/subdir` 只拷一部分，就会发现参数被无视。
3. 把仓库里真实的 `infra-bin/Dockerfile` 原文喂给 `docker build --check`，同样在 `COPY ${BAZEL_BIN} bazel-bin` 行报 `UndefinedVar`。
4. 注意 `--check` 只检查 Dockerfile 指令里的变量，RUN 命令字符串里的 `${BASE_IMAGE}`（由 shell 展开）不会报警，
   但它同样是空的（日志第一行 `BASE_IMAGE=''`）。
