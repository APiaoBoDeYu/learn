# lab06：Compose —— secrets、健康检查、网络、两个陷阱

对应第 6 章。结构仿照 `universal-uploader/assert/docker-compose.yaml`：

| 服务 | 模仿 | 要点 |
|---|---|---|
| `db` | uploader 的 `db` | `_FILE` secrets、`PGPORT=6432`、只绑 127.0.0.1、健康检查、**命名卷** |
| `app` | `universal_uploader` | `depends_on: condition: service_healthy`、`$$` 转义、从 secret 文件读密码、`init: true` |
| `app_naive` | （反例） | 只有 `depends_on: [db]`，没有健康检查和重启 |
| `app_hostnet` | `universal_uploader` 的 host 网络 | 经由宿主机端口映射访问 db |

"应用"直接借用 `postgres:16` 镜像里的 `psql`，不需要自己 build。

## 0. 准备

```bash
cd lab06_compose
./make_secrets.sh                 # 生成 secrets/pg_user、secrets/pg_password（同 deploy.sh）
docker compose config             # ★ 先看替换后的最终配置
```

在 `config` 的输出里找 `BUCKET_BAD`，看引号是否还在。实测：

```yaml
      BUCKET_BAD: '"demo-internal-data-us"'      ← 双引号成了值的一部分
      BUCKET_GOOD: demo-internal-data-us
```

## 1. 正常启动

```bash
docker compose up -d
docker compose ps                 # db 显示 (healthy)；app 在 db healthy 之后才启动
docker compose logs -f app        # 每 5 秒插入一行，rows 递增。Ctrl-C 退出 logs
```

**实测**：`up -d` 共 4 秒，输出里能看到 `lab06-db-1 Waiting → Healthy → lab06-app-1 Starting`。日志：

```
app-1  | BUCKET_BAD=["demo-internal-data-us"]     ← 值里带了双引号
app-1  | BUCKET_GOOD=[demo-internal-data-us]
app-1  | rows=1
app-1  | rows=2
```

这就是第 6 章说的 `MANAGED_DATA_S3_BUCKET="..."` 问题。

## 2. 服务名 DNS

`app` 里用 `psql -h db` 连接数据库 —— compose 为项目创建了一个网络（`lab06_default`），
同一网络内可以用**服务名**当主机名。

```bash
docker network ls | grep lab06
docker compose exec app getent hosts db       # 实测：172.18.0.2      db
```

## 3. host 网络 + 端口映射（uploader 的连接方式）

```bash
docker compose --profile hostnet up app_hostnet
```

**实测**：输出 `via host network: 172.18.0.2/32` —— 通过 `localhost:6432` 连上了，而服务端看到的地址是 db 容器在 bridge 网络里的 IP
（host 网络下的 localhost 是宿主机，再经 `127.0.0.1:6432:6432` 端口映射进 db 容器）。

## 4. 只有 `depends_on` 时会怎样

```bash
docker compose down -v                       # 先清空数据卷，让 Postgres 重新初始化（初始化要好几秒）
docker compose --profile naive up -d db app_naive
docker compose logs app_naive
```

**实测**：

```
app_naive-1  | psql: error: connection to server at "db" (172.18.0.2), port 6432 failed: Connection refused
app_naive-1  |  Is the server running on that host and accepting TCP/IP connections?
app_naive Exited (2)
```

而几秒后 db 已经是 `(healthy)`。
`depends_on` 只等 db **容器启动**，不等 Postgres **就绪**。
uploader 没出问题是因为 `restart: always` 让它反复重启直到成功。

再次运行 `docker compose --profile naive up app_naive`（此时 db 已就绪），就能连上了。

## 5. `down` vs `down -v`

```bash
docker compose up -d
sleep 20
docker compose logs --tail 1 app             # 实测：rows=4

docker compose down                          # 不带 -v
docker compose up -d && sleep 8
docker compose logs --tail 1 app             # 实测：rows=8，接着增长：命名卷保留了数据

docker compose down -v                       # 带 -v
docker compose up -d && sleep 8
docker compose logs --tail 1 app             # 实测：rows=2，重新开始：数据库被清空
```

这就是 `deploy.sh` 里 `down -v` 的风险。uploader 目前没出事，只是因为它的 db 用的是 bind mount
（`type: bind`，宿主机目录），`-v` 不删 bind mount。改成 bind mount 再做一遍试试：

```yaml
    volumes:
      - ./pgdata:/var/lib/postgresql/data
```

## 6. 停止耗时 —— 一个推翻了预测的结果

```bash
time docker compose stop app
```

**最初的 compose 文件里 app 只有 `init: true`，我预测 stop 会很快，实测却是 10.9 秒。** 排查过程：

```bash
docker inspect lab06-app-1 -f 'init={{.HostConfig.Init}} sig={{.Config.StopSignal}}'
# init=true sig=SIGINT              ← 停止信号是 SIGINT，不是 SIGTERM
docker image inspect postgres:16 -f '{{.Config.StopSignal}}'
# SIGINT                            ← 来自 postgres 镜像的 STOPSIGNAL，被继承了
```

再看进程的信号掩码（`/proc/<pid>/status`）：

```
1  /sbin/docker-init -- docker-entrypoint.sh bash -c ...
7  bash -c ...        SigCgt: 0000000000010002    ← bit 2 = SIGINT 被 bash 捕获
47 sleep 5            SigCgt: 0000000000000000
```

原因：
1. postgres 官方镜像设置了 `STOPSIGNAL SIGINT`（Postgres 的"快速关闭"信号），借用这个镜像的 app 服务继承了它。
2. tini 只把 SIGINT 转发给它的子进程 bash，不发给 `sleep`。
3. **非交互式 bash 收到 SIGINT 时，会等前台子进程结束，并且只有子进程也是被 SIGINT 杀死的才跟着退出**
   （bash 的 "wait and cooperative exit" 行为，为了让 `Ctrl-C` 在脚本里表现正确）。`sleep` 没收到信号、正常结束，bash 就当作没事继续循环。
4. 10 秒后 SIGKILL。

修复：在 app 上显式写 `stop_signal: SIGTERM`（当前 compose.yaml 已包含）。四种组合实测：

| stop_signal | init | stop 耗时 | ExitCode |
|---|---|---|---|
| SIGTERM | true | **0.8s** | 143 |
| SIGTERM | false | 10.9s | 137 |
| SIGINT（继承自镜像） | true | 10.9s | 137 |
| SIGINT（继承自镜像） | false | 10.9s | 137 |

教训：
- **`STOPSIGNAL` 会从基础镜像继承**，`docker inspect -f '{{.Config.StopSignal}}'` 检查一下你的镜像。
- `init: true` 解决的是"PID 1 不响应 SIGTERM"；信号本身选错了，init 也救不了。
- 这和 uploader 的 `finalize` 服务直接相关：它用的 `event-uploader` 基于 `python:3.11-slim`（没有设置 STOPSIGNAL，默认 SIGTERM），
  所以对它来说 `init: true` 就够了。

## 7. secrets 不在环境变量里

```bash
docker compose exec app env | grep -i pass             # 预期无输出：exec 起的是新进程，只继承容器配置的 Env；
                                                       # PGPASSWORD 只存在于 command 启动的那个 bash（init: true 时它不是 PID 1）及其子进程里
docker compose exec app sh -c 'grep -l PGPASSWORD /proc/[0-9]*/environ'   # 预期能找到：bash 的子进程（sleep、psql）继承了 export 的变量（/proc/<pid>/environ 只记录进程启动时的环境）
docker inspect lab06-app-1 -f '{{json .Config.Env}}'    # 容器配置里看不到密码
docker compose exec app ls -l /run/secrets/
```

实测：第一条输出为空；第二条只找到一个进程 `/proc/14/environ`，它的命令行是 `sleep 5`；`docker inspect` 的 Env 里没有密码；
`/run/secrets/` 下是 `pg_password`、`pg_user` 两个文件（就是宿主机 `secrets/` 目录里的文件挂进来的）。

对比：如果用 `environment: - PGPASSWORD=xxx`，`docker inspect` 里就能直接看到明文。

## 清理

```bash
docker compose --profile naive --profile hostnet down -v
rm -rf secrets pgdata
```
