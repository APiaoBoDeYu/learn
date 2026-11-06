# lab01：最小镜像，观察层与元数据

对应第 1 章。

## 步骤

```bash
cd lab01_basics

# 1. 构建，传入一个构建参数
docker build -t lab01:v1 --build-arg APP_VERSION=1.0.0 .

# 2. 看层：哪些指令产生了有大小的层？哪些是 0B？
docker history lab01:v1

# 3. 看元数据
docker image inspect -f '{{json .Config.Env}}'        lab01:v1
docker image inspect -f '{{.Config.WorkingDir}}'      lab01:v1
docker image inspect -f '{{.Config.Cmd}}'             lab01:v1
docker image inspect -f '{{json .Config.ExposedPorts}}' lab01:v1
docker image inspect -f '{{json .Config.Labels}}'     lab01:v1

# 4. 运行。注意：EXPOSE 不会映射端口，必须 -p
docker run -d --name lab01 -p 8000:8000 lab01:v1
curl localhost:8000
#    预期：cwd=/srv/app，APP_VERSION=1.0.0，BUILD_ONLY=null（ARG 不进运行期）

# 5. 构建期的 ARG 值写进了文件里（这是 ARG "泄漏"进镜像的一种方式）
docker exec lab01 cat /build-info.txt

# 6. 运行时覆盖 ENV
docker rm -f lab01
docker run -d --name lab01 -p 8000:8000 -e APP_VERSION=override lab01:v1
curl localhost:8000

# 7. 覆盖 CMD：镜像后面跟的东西替换整个 CMD
docker run --rm lab01:v1 python -c "import os; print(os.getcwd())"
docker run --rm -it lab01:v1 bash

# 8. 容器的可写层：在容器里写文件，用 docker diff 查看
docker exec lab01 sh -c 'echo hi > /tmp/x && pip --version > /srv/app/pip.txt'
docker diff lab01

docker rm -f lab01
```

## 实测结果（Jetson Thor, Docker 29.1.4）

`docker history lab01:v1`（节选）：

```
CMD ["python" "app.py"]                         0B
EXPOSE [8000/tcp]                               0B
COPY app.py . # buildkit                        856B
RUN |2 BUILD_ONLY=i-am-only-visible-during-b…   72B     ← RUN 那一行连 ARG 的值都记下来了
WORKDIR /srv/app                                0B
LABEL org.opencontainers.image.title=lab01 o…   0B
ENV APP_VERSION=1.0.0 PYTHONUNBUFFERED=1        0B
ARG APP_VERSION=1.0.0                           0B
ARG BUILD_ONLY=i-am-only-visible-during-build   0B
```

只有 `RUN` 和 `COPY` 产生了有大小的层，其余指令都是 0B（只改元数据）。

`curl localhost:8000`：

```json
{ "cwd": "/srv/app", "pid": 1, "APP_VERSION": "1.0.0", "BUILD_ONLY": null }
```

加 `-e APP_VERSION=override` 后 `APP_VERSION` 变为 `"override"`。

`docker history --no-trunc lab01:v1 | grep -c BUILD_ONLY` 输出 **2**（`ARG` 行和 `RUN` 行）——
**ARG 的值明文留在了镜像历史里**，这就是不能用 ARG 传密码的原因。

`docker diff lab01` 的意外收获：除了预期的 `A /tmp/x`、`A /srv/app/pip.txt`，还有 **405 个**
`A .../__pycache__/*.pyc` —— Python 运行时把字节码缓存写进了容器的可写层。
（生产镜像可以设 `PYTHONDONTWRITEBYTECODE=1`，或者在构建时预先编译好。）

思考题 3（把 CMD 改成 shell 形式 `CMD python app.py`）：

```
"pid": 7
PID 1 = /bin/sh -c python app.py        (/bin/sh -> dash)
```

**python-slim 里的 dash 没有做 exec 优化**，`/bin/sh` 是 PID 1，python 只是它的子进程（第 4 章的信号问题就出在这里）。
`docker build --check` 对这一行报 `JSONArgsRecommended: JSON arguments recommended for CMD to prevent unintended behavior related to OS signals`。

## 思考题

1. `docker history` 里 `ENV`、`WORKDIR`、`LABEL`、`CMD` 这几行大小是多少？为什么？
2. `ARG BUILD_ONLY` 的值能在 `docker history` 里看到吗？（试试 `docker history --no-trunc lab01:v1 | grep BUILD_ONLY`）
   这说明为什么不能用 ARG 传密码。
3. 把 `CMD ["python", "app.py"]` 改成 `CMD python app.py`，重新构建运行，`curl` 返回的 `pid` 有什么变化？
   （exec 形式下 python 是 PID 1；shell 形式下取决于 `/bin/sh` 是否做了 exec 优化，第 4 章详细讨论。）
