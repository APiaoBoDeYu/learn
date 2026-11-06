# 第 6 章：Docker Compose —— 多服务编排

当一个系统由多个容器组成（上传服务 + Postgres + 清理任务），每次手敲三条 `docker run` 既容易错也难维护。
Compose 把这些 `docker run` 参数写成一个 YAML 文件，一条命令启停全部。

**心智模型：compose 文件里的每个 `service` ≈ 一条 `docker run` 命令。** YAML 的每个字段几乎都能对应到第 5 章的某个参数。

## 6.1 v1 与 v2

| | Compose v1 | Compose v2 |
|---|---|---|
| 命令 | `docker-compose`（带连字符） | `docker compose`（空格，CLI 插件） |
| 实现 | Python | Go |
| 状态 | **2023 年 7 月停止维护** | 当前版本 |

仓库里两种都有：`docker-deploy/README.md` 写的是 `docker compose up -d`，而
`universal-uploader/deploy.sh` 检查并调用的是 `docker-compose`。新机器上一般只装了 v2，
`deploy.sh` 会直接报 `docker-compose command not found` 退出。建议统一到 v2。

文件开头的 `version: "3.7"` 在 v2 里**已经废弃**，会被忽略并给出警告，可以删掉。

## 6.2 常用命令

```bash
docker compose -f <文件> up -d             # 创建并后台启动所有服务
docker compose -f <文件> ps                # 状态
docker compose -f <文件> logs -f <服务>     # 日志
docker compose -f <文件> exec <服务> bash   # 进入容器（等价 docker exec）
docker compose -f <文件> restart <服务>
docker compose -f <文件> pull              # 拉取所有镜像的最新版本
docker compose -f <文件> down              # 停止并删除容器、网络
docker compose -f <文件> down -v           # ……同时删除命名卷和匿名卷 ⚠
docker compose -f <文件> config            # ★ 打印变量替换、合并之后的最终配置
```

**`docker compose config` 是排查 compose 问题的第一工具**：环境变量有没有替换成功、引号对不对、
路径是什么，一看便知。下面讨论的几个坑都可以用它验证。

不加 `-f` 时，默认读取当前目录下的 `compose.yaml` / `docker-compose.yaml` 等。

## 6.3 精读 `universal-uploader/assert/docker-compose.yaml`

这是仓库里最完整的 compose 文件：三个服务、secrets、两种网络模式、bind mount。

### 整体结构

```yaml
services:
    universal_uploader:              # 上传服务（主程序）
    uploader_finalize_orphaned_files: # 每 2 分钟清理一次孤儿文件
    db:                              # Postgres 16
secrets:                             # 顶层：定义 secret 从哪里来
    universal_uploader_postgres_user:
        file: ${LOCAL_DATA_DIR}/secrets/universal_uploader_postgres_user
    ...
volumes:
    shared-volume:                   # 定义了但没有任何服务使用（可删）
```

### 变量替换 `${VAR}`

```yaml
image: "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/universal-uploader:${VERSION}"
```

Compose 在解析文件时用**执行 compose 命令的 shell 环境**（以及同目录下的 `.env` 文件）替换 `${VAR}`。
这一步发生在**宿主机上、容器启动之前**。支持 `${VAR:-默认值}`、`${VAR:?报错信息}`。

`deploy.sh` 一开始检查一堆 `REQUIRED_ENV` 是否设置，就是因为这里引用的变量如果为空，
image 名会变成 `.dkr.ecr..amazonaws.com/universal-uploader:latest` 这种无效值。
更省事的写法是让 compose 自己检查：`${AWS_ACCOUNT_ID:?请先 source env.sh}`。

**注意区分**：`assert/config.yaml` 里也有 `${UNIVERSAL_UPLOADER_POSTGRES_USER}`、`${BUCKET_PREFIX}`，
但那个文件是被**挂载进容器、由应用程序自己**做变量替换的，和 compose 无关。

### `$$`：给容器里的 shell 留一个 `$`

```yaml
command: |
    bash -c "
        export UNIVERSAL_UPLOADER_POSTGRES_USER=$$(cat /run/secrets/universal_uploader_postgres_user) &&
        ...
```

compose 会把 `$(...)` 和 `$VAR` 当作自己的变量替换语法。写 `$$` 告诉 compose "这是字面量 `$`"，
传进容器后变成 `$(cat ...)`，由容器里的 bash 执行。

### environment：三种写法

```yaml
environment:
    - PYTHONUNBUFFERED=1                    # ① 固定值
    - AWS_ACCESS_KEY_ID                     # ② 只写名字：从宿主机环境透传（等价于 docker run -e AWS_ACCESS_KEY_ID）
    - RDP_URL=${PROD_ROBOT_DATA_BASE_URL}   # ③ 在宿主机上做变量替换后传入
    - MANAGED_DATA_S3_BUCKET="${BUCKET_PREFIX}-internal-data-us"   # ⚠
```

**⚠ 列表写法里的引号是值的一部分。** `- KEY="value"` 在 YAML 里是一整个字符串 `KEY="value"`，
compose 按第一个 `=` 切分，得到的值是带双引号的 `"xxx-internal-data-us"`。
应用拿这个值去访问 S3 bucket，bucket 名里多了两个引号字符。
lab06 实测，`docker compose config` 输出：

```yaml
      BUCKET_BAD: '"demo-internal-data-us"'
      BUCKET_GOOD: demo-internal-data-us
```

容器里打印出来是 `BUCKET_BAD=["demo-internal-data-us"]`。

正确写法：

```yaml
    - MANAGED_DATA_S3_BUCKET=${BUCKET_PREFIX}-internal-data-us      # 列表写法：不要引号
# 或者用字典写法，YAML 的引号会被正常解析掉：
environment:
    MANAGED_DATA_S3_BUCKET: "${BUCKET_PREFIX}-internal-data-us"
```

这个 bug 可能一直没暴露，是因为 `DEFAULT_FILE_MANAGER=local`，没有真正走到 S3 托管数据那条路径。

### secrets：比环境变量更安全地传密码

```yaml
services:
  db:
    environment:
      POSTGRES_USER_FILE: /run/secrets/universal_uploader_postgres_user
      POSTGRES_PASSWORD_FILE: /run/secrets/universal_uploader_postgres_password
    secrets:
      - universal_uploader_postgres_user
      - universal_uploader_postgres_password
secrets:
  universal_uploader_postgres_user:
    file: ${LOCAL_DATA_DIR}/secrets/universal_uploader_postgres_user
```

- 每个 secret 在容器里以文件形式出现在 `/run/secrets/<名字>`。
- 官方 postgres 镜像支持 `XXX_FILE` 约定：设置 `POSTGRES_PASSWORD_FILE` 后，从文件读取密码。很多官方镜像（mysql、redis 等）都支持这个约定。
- uploader 自己不支持 `_FILE` 约定，所以 `command` 里先 `cat` 文件再 export 成环境变量。
- `deploy.sh` 首次部署时用 `/dev/urandom` 生成 16 位随机密码写入文件。

比环境变量好在：不会出现在 `docker inspect` 的 Env 里，也不会被子进程意外打印。
（单机 compose 的 file secret 本质上是一个只读 bind mount，安全性来自"不进环境变量"，而非加密。）

### 网络：一个服务用 host，另一个用 bridge

```yaml
universal_uploader:
    network_mode: host                  # 宿主机网络
db:
    ports:
        - "6432:6432"                    # 没写 network_mode → 默认 bridge 网络 + 端口映射
    environment:
        PGPORT: 6432                     # 让 Postgres 在容器内也监听 6432（默认是 5432）
```

`config.yaml` 里 uploader 连的是 `postgresql://...@localhost:6432/uploader_data`。链路是：

```
uploader (host 网络，localhost = 宿主机)
   └─▶ 宿主机 localhost:6432 ──(docker 端口映射)──▶ db 容器内 :6432
```

两个注意点：
1. **`"6432:6432"` 默认绑定在 `0.0.0.0`**，同一局域网的任何机器都能连这个数据库。
   只需要本机访问时应写成 `"127.0.0.1:6432:6432"`。
2. 改了 `PGPORT` 之所以必要：如果只写 `"6432:5432"` 也能工作，但这里作者选择让内外端口一致，减少混淆。

### `depends_on` 不等于"等它就绪"

```yaml
universal_uploader:
    restart: always
    depends_on:
        - db
```

`depends_on` 只保证**先启动 db 容器，再启动 uploader 容器**。它不等 Postgres 真正可以接受连接
（Postgres 首次初始化数据目录可能要好几秒）。uploader 启动时连不上数据库就会崩溃退出 ——
这里能工作，靠的是 `restart: always` 让它不断重启，直到 db 就绪。

lab06 实测：在 db 首次初始化期间启动一个只有 `depends_on: [db]` 的客户端，结果是
`connection to server at "db" (172.18.0.2), port 6432 failed: Connection refused`，几秒后 db 才变成 healthy。

正规写法是给 db 加健康检查：

```yaml
db:
    healthcheck:
        test: ["CMD-SHELL", "pg_isready -p 6432 -U $$(cat /run/secrets/universal_uploader_postgres_user)"]
        interval: 5s
        timeout: 3s
        retries: 10
universal_uploader:
    depends_on:
        db:
            condition: service_healthy
```

### 清理服务：`pid: host` + `privileged`

```yaml
uploader_finalize_orphaned_files:
    image: ".../event-uploader:v6.3.0"
    # The finalize script needs to be able to list all open files, so it must run with the host
    # pid namespace in privileged mode
    pid: host
    privileged: true
    entrypoint: ["/bin/bash"]
    command: ["-c", "while true; do finalize_orphaned_files; sleep 120; done"]
```

注释写得很清楚：脚本要判断某个事件文件是否还被**宿主机上其他容器里的进程**打开着（类似 `lsof`），
只有加入宿主机的 PID namespace 才能看到其他容器的进程，只有 privileged 才有权限读取它们的 `/proc/<pid>/fd`。

问题：
- **这个服务没有 `restart:` 策略**，另外两个都是 `restart: always`。宿主机重启后它不会自动起来。
- `while true; sleep` 用 bash 做 PID 1，`docker stop` 要等 10 秒（第 4 章），可加 `init: true`。
  前提是停止信号是 SIGTERM：event-uploader 基于 `python:3.11-slim`，没有改 STOPSIGNAL，所以没问题；
  如果基础镜像带了 `STOPSIGNAL SIGINT`（比如 postgres），`init: true` 也没用，见第 4 章"STOPSIGNAL 会继承"。
- 用 cron 式的"容器内死循环"做定时任务是常见做法，简单有效。

### `container_name` 的取舍

`universal_uploader` 和 `db` 写死了 `container_name`。好处是名字固定，`docker logs universal-uploader` 很好记；
代价是同一台机器上**不能启动第二套**（名字冲突），也不能 `--scale`。部署在工作站上的单例服务，这样写合理。

不写 `container_name` 时，compose 生成的名字是 `<项目名>-<服务名>-<序号>`，
项目名默认是 compose 文件所在的**目录名**——这里是 `assert`，所以清理服务的容器会叫 `assert-uploader_finalize_orphaned_files-1`。
可以用 `docker compose -p universal-uploader ...` 或在文件顶部写 `name: universal-uploader` 指定。

## 6.4 精读 `deploy.sh`

`deploy.sh` 是一个典型的"compose 前置检查 + 启动"部署脚本：

1. 检查必需环境变量；
2. 检查/创建 `LOCAL_DATA_DIR` 下十几个目录（避免 bind mount 自动建出 root 所属的空目录，见第 5 章）；
3. 不存在就生成 secret 文件；
4. 首次部署时拷贝默认 `config.yaml`（已存在则不覆盖，保留现场修改）；
5. 从 compose 文件里 grep 出 image 名，检查本地是否已有这些镜像；
6. 停掉旧的，启动新的。

### 第 5 步的 `eval`

```bash
mapfile -t IMAGE_NAMES < <(grep '^\s*image:' "$ASSERT_DIR/docker-compose.yaml" | awk '{print $2}')
EVALUATED_IMAGE_NAME=$(eval echo "$IMAGE_NAME")
```

用 `eval` 来展开 `${AWS_ACCOUNT_ID}`。能用，但 `eval` 会执行字符串里的任意 shell 语法。
更稳的做法是让 compose 自己算：`docker compose -f ... config --images`。

### 第 6 步：一个永远为真的判断 + `down -v`

```bash
if docker ps -q --filter "name=universal-uploader" &> /dev/null; then
    echo "Stopping existing Docker container..."
    docker-compose -f "$ASSERT_DIR/docker-compose.yaml" down -v
fi
```

`docker ps -q --filter ...` **找没找到都返回退出码 0**（它只是输出为空；实测 `docker ps -q --filter name=definitely-nothing-here; echo $?` 输出 `0`），所以这个 if 永远成立。
想判断"有没有输出"要写成 `if [[ -n "$(docker ps -q --filter ...)" ]]`。
（不过 `down` 在没有容器时也是无害的，所以实际影响不大。）

**真正需要警惕的是 `down -v`**：它会删除 compose 文件里声明的**命名卷**和容器的匿名卷。
现在没出事，是因为 db 的数据用的是 bind mount（`type: bind`，宿主机目录），`-v` 不会删 bind mount。
但如果将来有人"优化"成命名卷：

```yaml
volumes:
  - pgdata:/var/lib/postgresql/data      # 命名卷
```

那**每次部署都会清空数据库**（lab06 实测：命名卷下 `down` 后行数从 4 继续增长到 8，`down -v` 后从头开始）。部署脚本里没有理由用 `-v`，应该去掉。

另外，`up -d` 本身就会对配置或镜像发生变化的服务做"停掉旧容器、创建新容器"，
通常不需要先 `down`。先 `down` 的副作用是所有服务都会有一段停机时间。

## 6.5 用户自定义网络：`video-monitoring/local-clipping-server`

```yaml
services:
    webserver_and_workers:
        networks:
            - video_local_clipping_network
        ports:
            - "5010:5010"
networks:
    video_local_clipping_network:
        external:
            name: video_local_clipping_network
```

- **用户自定义 bridge 网络**里的容器，可以用**服务名/容器名当主机名**互相访问（Docker 内置 DNS）。
  默认的 `bridge` 网络不支持这个，这是自定义网络的主要价值。
- `external: true` 表示这个网络不由本 compose 文件创建/删除，必须**事先**存在：
  ```bash
  docker network create video_local_clipping_network
  ```
  否则 `up` 报 `network video_local_clipping_network declared as external, but could not be found`。
  把网络声明为 external，通常是为了让**多个 compose 项目**共享同一个网络。
- `external: name: xxx` 是旧语法，新写法是：
  ```yaml
  networks:
    video_local_clipping_network:
      external: true
  ```

对比 host 网络：自定义网络保持了隔离，只把需要的端口（5010）暴露出来，是"服务器式"部署的推荐做法；
host 网络则是机器人现场"要直接碰硬件"时的务实选择。

## 6.6 模板化的 compose：`local-streaming-video`

```yaml
services:
  local_streaming_video-stream-<camera-id>:
    image: ".../video-monitoring-local-streaming:${VERSION}"
    network_mode: host
    entrypoint:
      ["python", "/opt/run_monitoring_service.py",
       "--stream-name", "video-stream-<camera-id>",
       "--rtsp-url", "rtsp://<camera-username>:<camera-password>@<camera-ip>:<camera-port>/cam/realmonitor?channel=1&subtype=0",
       ...]
```

这是一个**模板**，`<camera-id>` 等占位符要手动替换，每个摄像头复制一份 service。
- `entrypoint:` 用列表覆盖镜像的 ENTRYPOINT（参数都写在 entrypoint 里，而没有拆一部分到 command）。
- RTSP 密码明文写在 compose 文件里。同目录的 `configure_amcrest_camera.sh` 也把相机管理员密码硬编码在 curl 命令里。
  **凭据不应该提交进 git**，可以改成从环境变量（`${CAMERA_PASSWORD}`）或 secret 文件读取。

## 6.7 减少重复：扩展字段与 YAML 锚点

`covariant/infra/video_monitoring/docker-compose.yml`：

```yaml
x-volumes: &volumes                    # x- 开头的顶层字段：compose 忽略，专门用来放可复用片段
  - /var/covariant/app:/var/covariant/app
  - /var/covariant/log:/var/covariant/log

x-common-settings: &common             # &名字：YAML 锚点（定义）
  working_dir: /root/code
  network_mode: "host"
  restart: unless-stopped
  volumes: *volumes                    # *名字：YAML 别名（引用）

services:
  some_service:
    <<: *common                        # 合并键：把 common 的所有字段展开到这里
    image: ...
```

服务很多、配置大同小异时非常好用。锚点是 **YAML 语言**本身的特性，不是 compose 的；`x-` 前缀是 compose 的约定。

## 6.8 其他值得知道的特性

| 特性 | 用途 |
|---|---|
| 多文件合并 `-f base.yml -f override.yml` | 开发/生产差异化配置；`docker-compose.override.yml` 会被自动加载 |
| `profiles: [debug]` | 某些服务只在 `--profile debug` 时启动 |
| `env_file: .env.prod` | 从文件给容器注入环境变量（注意和"用 .env 做变量替换"是两回事） |
| `deploy.resources.limits` | 资源限制 |
| `logging: driver/options` | 日志轮转（`max-size: 100m`），**长期运行的工作站服务强烈建议配置**，否则 json 日志会写满磁盘 |
| `init: true` | 用 tini 做 PID 1（第 4 章） |
| `stop_grace_period: 30s` | `docker stop` 的等待时间 |

## 6.9 动手

做 [code/lab06_compose](code/lab06_compose/)：复刻 uploader 的结构（应用 + Postgres + secrets + 健康检查），
并亲手验证 `depends_on` 的问题、`environment` 引号问题和 `down -v` 的危险。

下一章：[07_tag_push_multiarch.md](07_tag_push_multiarch.md)
