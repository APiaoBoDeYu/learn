# 第 2 章：构建上下文、层缓存与镜像体积

写出"能用"的 Dockerfile 不难，写出"构建快、镜像小、不出怪问题"的 Dockerfile，靠的就是本章三件事。

## 2.1 构建上下文（build context）

### 是什么

```bash
docker build -t myimg -f path/to/Dockerfile  <上下文路径>
#                       └─ Dockerfile 位置    └─ 这个目录会被打包发给 daemon
```

- `COPY`/`ADD` 的源路径**全部相对于上下文根目录**，与 Dockerfile 放在哪无关。
- **上下文之外的文件，`COPY` 永远拿不到**。`COPY ../xxx` 会报错（越界）。
- `-f` 指定 Dockerfile 的位置，它可以在上下文之外。

为什么要有"上下文"这个概念？因为 CLI 和 daemon 是分开的（甚至可以在不同机器上，
CI 里用的 `docker -H tcp://...` 就是连远程 daemon），daemon 看不到你本地的文件系统，
只能看到 CLI 打包发过去的那部分。

### 仓库里四种不同的上下文策略

**策略 1：上下文 = 仓库上一级目录（brain-service）**

`build_docker.sh`：

```bash
current_file_dir=".../docker-deploy/brain-service/scripts"
source_dir="$(dirname "$(dirname "$current_file_dir")")"   # .../docker-deploy
work_dir="$(realpath "$source_dir/../..")"                  # /root/brain
cd "$work_dir"
docker build ... --file=./brain-service/docker-deploy/brain-service/Dockerfile_${arch} ... .
```

所以上下文是 `/root/brain`，它下面有：

```
/root/brain/
├── .git/            ← 198 MB（本机实测）
├── bamboo-specs/
├── source-deploy/
├── brain-service/   ← 197 MB（本机实测，不含 bazel 输出）
└── fe_files/        ← CI 里由 bamboo 脚本临时 cp 进来（见 bamboo.yaml）
```

为什么要把上下文放到上一级？因为 Dockerfile 要 `COPY fe_files ...`，而 `fe_files` 不在
`brain-service/` 里面 —— CI 先把 `/var/covariant/managed_data/fe_files` 拷到工作区根目录，
再 build。**上下文必须是"所有要 COPY 的东西"的公共祖先目录。**

代价：`/root/brain` 下没有 `.dockerignore`，`.git` 也在上下文里。
旧版构建器（legacy builder）会把整个上下文打包发送，日志里会看到
`Sending build context to Docker daemon  xxx MB`；BuildKit 按需传输、并在多次构建间做增量同步，
所以没用到的 `.git` 影响较小 —— 但 `COPY brain-service .` 这条会把 `brain-service/`
下的**一切**都拷进镜像，包括本地的临时文件、`__pycache__`、本地装的 `node_modules` 等，
这些东西还会让缓存频繁失效（见 2.2）。

**策略 2：上下文 = 一个专门准备好的小目录（infra-bin / model-bin / universal-uploader）**

`infra-bin/build.sh`：

```bash
bazel build covariant/infra/bin/...

# Copy only the bazel bin and entrypoint script into the docker build
tar -hcf bazel-bin.tgz bazel-bin/*     # ← -h：跟随软链接，打包实际文件
rm -rf bazel-bin                       # ← 删掉 bazel-bin 这个软链接本身
tar -xf bazel-bin.tgz                  # ← 解出一个实体目录 bazel-bin/
cp $SCRIPT_DIR/entrypoint.sh bazel-bin/entrypoint.sh

# Build from inside the bazel-bin directory to limit the docker context size
(cd bazel-bin; docker build ... -f $SCRIPT_DIR/Dockerfile .)
```

这一段的每一步都有理由，值得仔细理解：

1. **为什么要 `tar -h`？** 本机上 `brain-service/bazel-bin` 是这样一个软链接：
   ```
   bazel-bin -> /root/.cache/bazel/_bazel_root/geekplus/execroot/covariant_ws/bazel-out/aarch64-opt/bin
   ```
   **Docker 的 COPY 不跟随软链接**，会把软链接原样拷进镜像。而它指向的 `/root/.cache/...`
   在上下文之外、在镜像里也不存在，于是镜像里得到一个悬空链接。实测（在上下文里放一个指向外部目录的 `bazel-bin` 链接，然后 `COPY . /c`）：
   ```
   lrwxrwxrwx  bazel-bin -> /tmp/s/outside
   ls: /c/bazel-bin/: No such file or directory
   ```
   指向上下文**内部**的软链接（如 `inner-link -> sub`）则可以正常使用。bazel 的 runfiles 目录内部也大量使用软链接。
   `tar -h` 把所有链接替换成真实文件，问题就解决了。
2. **为什么 `cd bazel-bin` 再 build？** 让上下文只包含产物，几百 MB 的源码一个字节都不发。
3. **为什么 entrypoint.sh 要先 cp 进 bazel-bin？** 因为 Dockerfile 里有 `COPY entrypoint.sh entrypoint.sh`，
   它必须在上下文里。

**策略 3：临时把上下文外的东西拷进来，build 完再删（robot-data-platform、event-uploader）**

```bash
# robot-data-platform/build.sh
cp -r $RDP_WS/../rdp_shared_structs $RDP_WS/temp_rdp_shared_structs
(cd $RDP_WS; docker build ... .)
rm -rf $RDP_WS/temp_rdp_shared_structs
```

```bash
# event-uploader/build.sh
cp -r $cov_libs_path $script_dir/deps/
docker build -t $image_name ... $script_dir
rm -rf $script_dir/deps/*
```

`event-uploader/deps/.gitignore` 里写着 `cov_libs/`，防止这个临时副本被误提交。

这是绕开"上下文之外拿不到"限制的老办法。缺点：脚本中途失败（`set -e` 退出）时临时目录不会被清理。
**BuildKit 的新办法是"命名上下文"**，可以给一次构建传多个上下文：

```bash
docker buildx build \
  --build-context shared=../rdp_shared_structs \
  -f Dockerfile .
```

```dockerfile
COPY --from=shared . /app/rdp_shared_structs
```

不需要临时拷贝，也不会污染源码目录。

**策略 4：`.dockerignore`**

`covariant/infra/robot_data_platform/.dockerignore`：

```
/env
/pg-data
__pycache__
```

和 `.gitignore` 语法类似，放在**上下文根目录**，被匹配的文件不进入上下文。
这里排除了本地 venv、本地 Postgres 数据目录 —— 如果不排除，开发者本机跑过一次服务，
就会把几百 MB 的数据库文件拷进镜像。

一个典型的 Python/bazel 项目 `.dockerignore` 起点：

```
.git
**/__pycache__
**/*.pyc
**/node_modules
bazel-*
*.tar
*.tgz
.vscode
.idea
```

### 验证上下文里到底有什么

```bash
# 用一个只 COPY 全部上下文的临时 Dockerfile 来看
printf 'FROM busybox\nCOPY . /ctx\nCMD find /ctx -maxdepth 2\n' | \
  docker build -t ctx-check -f - . && docker run --rm ctx-check
```

## 2.2 层缓存

### 缓存命中规则

docker build 逐条处理指令，每一条都先查缓存：

| 指令 | 缓存命中条件 |
|---|---|
| `FROM` | 本地已有这个镜像（**不会**自动检查仓库里是否有更新的同名 tag，除非加 `--pull`） |
| `RUN` | 父层相同 **且 指令字符串完全相同** |
| `COPY` / `ADD` | 父层相同 且 指令相同 **且 被复制文件的内容（校验和）相同** |
| `ENV` / `ARG` / `WORKDIR` 等 | 父层相同 且 指令（含变量展开后的值）相同 |

**最关键的一条：某一层缓存失效，它之后的所有层都必须重建。**

所以写 Dockerfile 的第一原则：**把变化少的放前面，变化多的放后面**。

### `RUN` 的缓存不知道外部世界变了

`RUN` 只比较字符串。C 文件（universal-uploader）里：

```dockerfile
RUN curl -fsSL -o ~/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && ...
```

URL 里是 `latest`。缓存命中后，即使 Miniforge 出了新版本，你得到的永远是第一次 build 时的那个。
反过来，一旦缓存失效（比如换了台机器），就会拿到一个**不同的版本** —— 构建不可复现。
同理 `apt-get update`、`pip install 不带版本号的包`。

解决思路：
- 把版本钉死（`Miniforge3-24.3.0-0-Linux-x86_64.sh`、requirements 写 `==`）。
- 需要强制刷新时，用 `--no-cache`（全部重建）或 `--no-cache-filter <阶段名>`（BuildKit，只重建某阶段）。

这就是为什么仓库里 infra-bin、model-bin、universal-uploader、rdp 的 build.sh 都带 `--no-cache`：
宁可每次全量重建，也不要拿到"不知道多旧"的缓存层。这是一个合理但昂贵的取舍。

### 正例：robot-data-platform

```dockerfile
FROM .../python:3.11
COPY requirements/base.txt /app/base.txt              # ① 依赖清单（很少变）
COPY requirements/local.txt /app/requirements.txt
RUN ... pip install -r /app/requirements.txt          # ② 装依赖（慢，几分钟）
COPY robot_data_platform /app/robot_data_platform     # ③ 源码（天天变）
COPY ...
```

改了源码 → 只有 ③ 及之后重建，② 那个最慢的步骤命中缓存。
改了 requirements → 从 ① 开始重建。
这就是"**先拷依赖清单，装依赖，再拷源码**"的经典模式，所有语言通用（npm 是先拷 `package.json` + `yarn.lock`）。

顺便说一个细节：为什么要单独 `COPY requirements/base.txt /app/base.txt`？
因为 `local.txt` 里很可能写着 `-r base.txt`，pip 会相对于 requirements 文件所在目录去找它，
所以两个文件必须放在同一个目录（`/app/`）下。

（如果 rdp 的 build.sh 不加 `--no-cache`，这个顺序带来的收益才能真正体现出来。）

### 反例：brain-service

```dockerfile
COPY brain-service .                                     # ① 整个源码树
COPY fe_files /var/covariant/managed_data/fe_files/
RUN cp -r /opt/geekplus/third_party /root/brain-service/
RUN cp -r /opt/geekplus/node_modules /root/brain-service/covariant/gui/
RUN cp -r /opt/geekplus/ui_node_modules/node_modules /root/brain-service/covariant/gui/ui/
RUN bazel build $bazel_options //covariant/service:launch   # ② 最慢的一步
RUN cd covariant/gui && yarn && yarn build && cd -         # ③
```

任何一个源文件变化 → ① 失效 → 后面全部重建，包括：
- 三次 `cp -r`（与源码无关，却每次重做）；
- **从零开始的 bazel build** —— bazel 自己的增量缓存在 `/root/.cache/bazel`，
  但上一次构建的那份缓存在上一个镜像的层里，新构建用不上。

另外两个影响体积的问题：
- `cp -r /opt/geekplus/third_party ...`：基础镜像里已经有一份，这里再复制一份，**镜像里同样内容存了两遍**。
  如果 bazel 的 WORKSPACE 能直接引用 `/opt/geekplus/third_party`（或者用软链接 `ln -s`），可以省掉。
- bazel 的整个 output base（含所有外部依赖、中间产物）都留在 `/root/.cache/bazel` 这一层里，被打进了生产镜像。

改进方向（第 8 章会给出完整改写）：

1. **BuildKit 缓存挂载**，让 bazel 缓存跨构建保留、且不进镜像：

   ```dockerfile
   RUN --mount=type=cache,target=/root/.cache/bazel \
       bazel build //covariant/service:launch && \
       cp -rL bazel-bin/covariant/service/launch* /opt/brain-out/
   ```

   ⚠ 注意 `cp -rL`：缓存挂载的内容**不会进镜像层**，而 `bazel-bin` 指向的正是缓存目录里的文件。
   如果不在同一个 RUN 里把产物拷出来（`-L` 跟随软链接），构建完镜像里就只剩悬空链接。

2. **多阶段构建**：在 builder 阶段编译，最终镜像只拿产物（第 3 章）。

### 用 `--progress=plain` 观察缓存

`build_docker.sh` 用了 `--progress=plain`，构建日志会逐行展开，命中缓存的步骤显示 `CACHED`：

```
#5 [2/8] WORKDIR /root/brain-service
#5 CACHED
#6 [3/8] COPY brain-service .
#6 DONE 4.2s
```

CI 日志排错时非常有用。本地交互式构建默认是 `--progress=auto`（折叠的进度条）。

## 2.3 镜像体积

### 先测量，再优化

```bash
docker images                               # 总大小
docker history --no-trunc <镜像>             # 每层大小与产生它的指令
docker run --rm <镜像> du -sh /root/.cache /opt/* 2>/dev/null   # 看具体目录
```

第三方工具 [dive](https://github.com/wagoodman/dive) 可以交互式地逐层浏览文件变化，强烈推荐。

### 仓库里能看到的体积优化手法

| 手法 | 例子 |
|---|---|
| 小基础镜像 | `event-uploader` 用 `python:3.11-slim`（约 130MB），而 `robot-data-platform` 用完整的 `python:3.11`（约 1GB） |
| apt 装完立即清索引（同一个 RUN） | `&& rm -rf /var/lib/apt/lists/*`（universal-uploader、rdp、event-uploader） |
| `--no-install-recommends` | universal-uploader、rdp |
| pip 不留缓存 | `pip install --no-cache-dir`（rdp、event-uploader） |
| 安装用的源文件装完就删（同一个 RUN） | `event-uploader`：`pip install deps/cov_libs && ... && rm -rf /root/deps`；`local_clipping`：下载的 ffmpeg tar 包解压后立即删 |
| 编译工具用完卸载（同一个 RUN） | `local_clipping` production 阶段：`pip install ... && apt-get purge -y --auto-remove gcc xz-utils` |

### 仓库里的体积问题

1. **rdp：在官方 python 镜像里 `apt-get install python3-pip`**。官方 `python:3.11` 镜像
   自带 `/usr/local/bin/python` 和 pip；而 apt 的 `python3-pip` 依赖 Debian 系统自带的 `python3`，
   相当于**又装了一个 Python 解释器**进 `/usr/bin`。后面 `python -m venv /env` 用的是 `/usr/local/bin/python`，
   apt 装的那个根本没用上。这一行可以整个删掉。
2. **event-uploader：装了 gcc 没卸**（`cov_libs` 编译 C 扩展需要 gcc，装完就没用了）。
   可以像 `local_clipping` 那样在同一个 RUN 里 purge，或用多阶段构建。
3. **event-uploader：注释里的 `[cite: 176]`** —— 看起来是从 AI 对话里复制来的残留，与体积无关，但说明这个文件没经过仔细 review。
4. **brain-service：源码 + 完整工具链 + bazel 缓存 + 重复的 third_party 全在生产镜像里**（见上文）。

### "删了但没变小"

```dockerfile
RUN wget big.tar.xz
RUN tar xf big.tar.xz && rm big.tar.xz     # ✗ big.tar.xz 已经永久留在上一层
```

```dockerfile
RUN wget big.tar.xz && tar xf big.tar.xz && rm big.tar.xz   # ✓ 一层内完成
```

`local_clipping/Dockerfile` 下载 ffmpeg 的 RUN 就是正确写法：下载、解压、移动、删除全在一个 RUN 里。

## 2.4 动手

做 [code/lab02_cache](code/lab02_cache/)：同一个项目两种写法，改一行源码后对比重建耗时，
并亲手体验 "删了但没变小"。

下一章：[03_multistage_and_targets.md](03_multistage_and_targets.md)
