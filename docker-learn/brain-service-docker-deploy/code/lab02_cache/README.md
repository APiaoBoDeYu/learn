# lab02：层缓存与镜像体积

对应第 2 章。

## A. 指令顺序与缓存

```bash
cd lab02_cache
./run.sh
```

脚本会：冷构建两个版本 → 改一行 `main.py` → 再构建两个版本并计时。

**实测**（Jetson Thor，pip 走外网）：

```
== 第一次构建（冷缓存，两者都要 pip install）
Dockerfile.bad       7.87s
Dockerfile.good      5.64s
== 修改一行源码后重建
Dockerfile.bad       5.46s      ← 又装了一遍依赖
Dockerfile.good      1.25s      ← pip 那一步 CACHED
== 用 --progress=plain 看 good 版本哪些步骤是 CACHED
#6 [3/5] COPY requirements.txt .
#6 CACHED
#7 [2/5] WORKDIR /app
#7 CACHED
#8 [4/5] RUN pip install --no-cache-dir -r requirements.txt
#8 CACHED
#9 [5/5] COPY . .
```

这里只有两个小依赖，差距是 4 倍；依赖越重（torch、bazel 外部依赖）差距越大。

这就是 `robot-data-platform/Dockerfile`（先拷 requirements）和 `brain-service/Dockerfile_*`（先拷全部源码）的区别。

注意：构建上下文是 `app/` 目录（`docker build ... app`），而 Dockerfile 在 `app/` 外面 ——
`-f` 指定的 Dockerfile 可以不在上下文里。

## B. .dockerignore

```bash
# 往上下文里放一个 200MB 的文件
head -c 200M /dev/urandom > app/big.bin

# 有 .dockerignore（里面排除了 big.bin）
docker build -f Dockerfile.bad -t lab02:bad app
docker run --rm lab02:bad ls -la /app        # 实测：只有 main.py、requirements.txt；镜像 167MB

# 去掉 .dockerignore 再试
mv app/.dockerignore app/.dockerignore.off
docker build -f Dockerfile.bad -t lab02:bad app
docker run --rm lab02:bad ls -la /app        # 实测：多了 big.bin
docker images lab02:bad                      # 实测：377MB（+210MB，即 200MiB）

mv app/.dockerignore.off app/.dockerignore
rm app/big.bin
```

## C. 删了但没变小

```bash
docker build -f Dockerfile.layers_bad  -t lab02:layers_bad  app
docker build -f Dockerfile.layers_good -t lab02:layers_good app
docker images | grep lab02
docker history lab02:layers_bad
```

**实测**：`layers_good` 150MB，`layers_bad` **255MB**（多 105MB）。`docker history lab02:layers_bad`：

```
RUN /bin/sh -c rm /big.bin # buildkit           0B
RUN /bin/sh -c head -c 100M /dev/urandom > /…   105MB
```

`rm` 只是在新层里标记删除，下层的 105MB 还在。

## D. RUN 的缓存不知道外部世界

```bash
cat > /tmp/Dockerfile.date <<'X'
FROM busybox
RUN date > /built-at
X
docker build -t lab02:date -f /tmp/Dockerfile.date /tmp/
docker run --rm lab02:date cat /built-at
sleep 5
docker build -t lab02:date -f /tmp/Dockerfile.date /tmp/
docker run --rm lab02:date cat /built-at       # 实测：02:23:00，时间没变——RUN 命中了缓存
docker build --no-cache -t lab02:date -f /tmp/Dockerfile.date /tmp/
docker run --rm lab02:date cat /built-at       # 实测：02:23:09，时间更新了
```

同样的道理：`RUN curl .../latest/...`、`RUN apt-get update` 命中缓存时，拿到的是第一次构建时的内容。
