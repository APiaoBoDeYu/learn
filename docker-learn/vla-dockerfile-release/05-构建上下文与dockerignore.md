# 05 构建上下文与 .dockerignore

> 另见：[brain-service 篇第 2 章 2.1](../brain-service-docker-deploy/02_context_and_cache.md)

## 为什么需要 .dockerignore

构建上下文目录里的文件都可能被发给构建引擎。原文件的上下文是整个仓库，里面有模型权重、数据集、日志、`.git`，动辄几十 GB。不过滤的话：

- 构建前传输上下文很慢。
- `COPY` 一个目录时，可能把权重、`.env` 里的密钥一起拷进镜像。

`.dockerignore` 列出的文件不会进入上下文，`COPY` 也就拷不到它们。

## 语法

和 `.gitignore` 很像：

```
# 注释
*.log            # 上下文根目录下的 .log 文件
**/*.log         # 任意深度的 .log 文件
**/__pycache__   # 任意深度的 __pycache__ 目录
!keep.log        # ! 表示例外：重新放行
```

- 规则从上到下依次匹配，**后面的规则覆盖前面的**。
- 路径相对于上下文根目录。

## 白名单写法

原文件用的是先全部排除、再逐个放行的写法：

```
*
!brain/vla/openpi
!body/gino

**/checkpoints
**/.git
**/.env
...
```

执行顺序：

1. `*` 排除所有文件。
2. `!brain/vla/openpi` 和 `!body/gino` 放行这两个目录及其内容。
3. 再在放行的目录里排除 checkpoints、缓存、密钥等。

和黑名单写法（只列要排除的）相比，白名单更安全：仓库以后新增的目录默认不会进镜像，不会因为忘了加排除规则而泄露数据。

## 每个 Dockerfile 一份 ignore 文件

默认情况下，Docker 读取**上下文根目录**下的 `.dockerignore`。

BuildKit 还支持另一种写法：在 Dockerfile 旁边放一个 `<Dockerfile 文件名>.dockerignore`，它的优先级更高。

```
docker-deploy/vla/
├── Dockerfile.release
└── Dockerfile.release.dockerignore     ← 构建 Dockerfile.release 时自动使用
```

这样同一个仓库里的不同 Dockerfile 可以用各自的过滤规则，互不干扰。

注意两点：

- 规则里的路径仍然相对于**上下文根目录**（仓库根），不是相对于 ignore 文件所在的目录。
- 这是 BuildKit 的功能。关掉 BuildKit 用旧构建器时，这个文件不会生效。

## 动手试一试

```bash
mkdir -p ~/docker-lab/ch05/{app,data,secret} && cd ~/docker-lab/ch05
echo 'print(1)' > app/main.py
echo 'big'      > data/weights.bin
echo 'KEY=xxx'  > secret/.env

cat > Dockerfile <<'EOF'
FROM python:3.12-slim
COPY . /ctx
RUN find /ctx -type f
EOF

docker build --progress=plain --no-cache -t lab:ch05 .        # 能看到所有文件

cat > Dockerfile.dockerignore <<'EOF'
*
!app
EOF

docker build --progress=plain --no-cache -t lab:ch05 .        # 只剩 app/main.py
```

`--no-cache` 强制重新执行每一步，确保看到最新的 `find` 结果。Dockerfile 本身被排除也不影响构建，构建引擎会单独读取它。
