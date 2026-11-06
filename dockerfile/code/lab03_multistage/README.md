# lab03：多阶段构建与 --target

对应第 3 章。

## 步骤

```bash
cd lab03_multistage

# 1. 单阶段
docker build -f Dockerfile.single -t lab03:single .

# 2. 多阶段：不指定 --target 时构建最后一个阶段（runtime-scratch）
docker build -f Dockerfile.multi -t lab03:scratch .

# 3. 用 --target 构建其他阶段
docker build -f Dockerfile.multi --target runtime-slim -t lab03:slim .
docker build -f Dockerfile.multi --target dev          -t lab03:dev  .

# 4. 对比体积
docker images lab03

# 5. 运行
docker run --rm lab03:single
docker run --rm lab03:slim
docker run --rm lab03:scratch
docker run --rm lab03:scratch --help         # ENTRYPOINT 固定，参数追加给 /hello

# 6. scratch 镜像里什么都没有
docker run --rm --entrypoint sh lab03:scratch     # 报错退出：scratch 里没有 sh

# 7. dev 镜像：把源码挂进去改代码、重新编译，不需要重建镜像
docker run --rm -it -v "$PWD":/src lab03:dev bash -c "gcc -g -o /tmp/h hello.c && /tmp/h"
```

## 实测（Jetson Thor, arm64）

| 镜像 | 体积 | 里面有什么 |
|---|---|---|
| `debian:bookworm-slim`（参照） | 97.2MB | 基础镜像 |
| `lab03:single` | 275MB | debian + gcc + libc6-dev + 源码 + 产物 |
| `lab03:dev` | 364MB | 同上 + gdb + make |
| `lab03:slim` | 98MB | debian-slim + 一个静态二进制（只比基础镜像多 0.8MB） |
| `lab03:scratch` | **703kB** | 只有一个静态二进制 |

四个镜像运行结果都是 `hello from pid 1`。同一份程序，镜像体积相差近 400 倍。

## 观察 BuildKit 只构建需要的阶段

```bash
docker build -f Dockerfile.multi --target runtime-slim --progress=plain --no-cache . 2>&1 | grep -E '^#[0-9]+ \['
```

**实测**：日志里只有 `[builder 1/5]`～`[builder 5/5]` 和 `[runtime-slim 2/2]`，没有任何 `dev` 阶段的步骤 ——
BuildKit 只构建目标阶段依赖链上的阶段。
旧的 legacy builder 会把目标之前的所有阶段都构建一遍。

## 思考题

1. `runtime-scratch` 为什么必须 `-static` 链接？（scratch 里没有 libc.so）
2. 为什么 Python 程序一般不能用 scratch？（需要解释器和一大堆 .so）
3. brain-service 要改成多阶段，runtime 阶段该选什么基础镜像？需要从 builder 拷哪些东西？
   bazel runfiles 里的软链接要怎么处理？（第 3 章 3.6）
