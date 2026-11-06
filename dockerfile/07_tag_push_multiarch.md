# 第 7 章：命名、推送、离线分发与多架构镜像

镜像造好之后，要让它出现在 CI、开发机、Jetson 工作站上。本章对照各个 `build.sh` 的后半段。

## 7.1 tag 只是一个指针

```bash
docker build -t brain-service:latest -t brain-service:v20250907_085100_UTC .
```

一个镜像可以有任意多个 tag，它们指向同一个镜像 ID（同一组层）。`docker tag` 只是**加一个新名字**，不复制任何数据：

```bash
docker tag brain-service:v1.6.2_x86_64 robot-harbor.geekplus.com/gino/brain-service:v1.6.2_x86_64
```

推送到哪个仓库，完全由 tag 里的 registry 部分决定 —— 所以推送前必须先打一个带仓库地址的 tag。

### 仓库里的三种 tag 策略

| 来源 | 格式 | 例子 |
|---|---|---|
| 各 build.sh 默认 | `latest` + UTC 时间戳 | `infra-bin:latest`、`infra-bin:v20250907_085100_UTC` |
| `build_docker.sh` 默认 | 架构前缀 | `brain-service:arm64_latest`、`brain-service:arm64_v20250907_...` |
| CI（`branch_check.sh`） | 语义版本 + 架构后缀；非主干分支加 `rc` | `brain-service:v1.6.2_x86_64`、`v1.6.2rc_aarch64` |

`branch_check.sh` 里版本号是**手写在脚本里**的（`BRAIN_SERVICE_VERSION=v1.6.2`），CI 通过对比
`HEAD^` 上的版本号来判断"这个服务的版本变了没有、要不要构建"。改版本号 = 触发发版。

**为什么部署时不该用 `latest`**：`latest` 不是"最新"的意思，只是一个默认 tag 名，它会被每次构建覆盖。
工作站上 `docker pull xxx:latest` 拿到的是什么版本、和昨天是否相同，无从得知；出了问题也无法回滚到"上一个 latest"。
compose 文件里用 `${VERSION}` 而不是写死 `latest`，就是为了部署时明确指定版本。

## 7.2 登录与推送

### AWS ECR

```bash
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$AWS_DOCKER_REGISTRY"
docker tag  "$IMAGE:$TAG" "$AWS_DOCKER_REGISTRY/$IMAGE:$TAG"
docker push "$AWS_DOCKER_REGISTRY/$IMAGE:$TAG"
```

- ECR 的密码是 AWS CLI 用 IAM 凭证换来的**临时 token（12 小时有效）**，所以每个脚本都要先登录一次。
- `--password-stdin` 让密码从管道传入，不会出现在进程列表和 shell 历史里。✅

### Harbor（公司内部仓库）

```bash
docker -H tcp://$GEEK_DOCKER_IP login robot-harbor.geekplus.com -u $GEEK_DOCKER_USER -p $GEEK_DOCKER_KEY
docker tag  "$IMAGE:$TAG" "$GEEK_DOCKER_REGISTRY/$IMAGE:$TAG"
docker push "$GEEK_DOCKER_REGISTRY/$IMAGE:$TAG"
```

两个值得注意的地方：

1. **`-p $GEEK_DOCKER_KEY`**：密码作为命令行参数，在执行期间对 `ps` 可见，docker 自己也会警告
   `Using --password via the CLI is insecure`。应改为
   `echo "$GEEK_DOCKER_KEY" | docker login ... -u "$GEEK_DOCKER_USER" --password-stdin`。
2. **`-H tcp://$GEEK_DOCKER_IP`**：只有 login 这一条命令连到了一个**远程** daemon，后面的 push 连的是本地 daemon。
   login 的凭据保存在**客户端**的 `~/.docker/config.json` 里，所以后续本地 push 仍能使用 **【待验证】**。
   为什么要经由远程 daemon 登录，从代码里看不出原因，可能是历史遗留。另外 `tcp://` 不带 TLS 的 daemon 端口本身就是一个严重的安全隐患（等同于开放 root）。

### 推送的是层

`docker push` 逐层上传，仓库里已有的层（按内容哈希判断）显示 `Layer already exists`，直接跳过。
所以 infra-bin、model-bin 这些基于 `brain-service-dev` 的薄镜像，push 时只传自己新增的那几层。

### 同一个镜像推两个仓库

每个脚本都是先推 ECR、再推 Harbor，相同内容传两遍。替代方案：
- Harbor 自带**仓库复制（replication）**功能，可以配置从 ECR 自动同步；
- 用 `skopeo copy` / `crane copy` 在仓库之间直接复制，不经过本地 daemon。

## 7.3 离线分发：`docker save` / `docker load`

工作站可能不能访问外网或仓库。`docker-deploy/README.md` 描述的流程：

```bash
./brain-service/docker-deploy/brain-service/scripts/build_docker.sh --export   # 构建并导出 tar
# 拷贝到工作站（U 盘 / scp）
docker load -i docker/brain-service.docker-image.tar
docker compose up -d
```

- `docker save` 把镜像（**所有层 + 元数据 + tag**）打成一个 tar，`docker load` 原样恢复。
- 可以一次保存多个镜像：`docker save img1:t img2:t -o all.tar`，共享的层只存一份。
- tar 包通常压缩效果不错：`docker save img | gzip > img.tar.gz`；`docker load` 能直接读 gzip。

**不要和 `docker export` / `docker import` 混淆**：export 导出的是一个**容器**的文件系统，
压平成一层，丢失所有元数据（ENTRYPOINT、ENV、WORKDIR……）。分发镜像用 save/load。

### ⚠ `--export` 在当前代码里不能用

`build_docker.sh` 最后：

```bash
if [[ "$export_image" == true ]]; then
    mkdir -p $exported_dir
    docker save "$tag" -o $exported_dir/brain-service.docker-image.tar
fi
```

整个脚本里**没有定义过 `$tag` 变量**（定义的是 `IMAGE` 和数组 `TAGS`），这里展开成 `docker save ""`，
实测报错 `Error response from daemon: invalid reference format`。由于脚本有 `set -e`，`--export` 会以失败告终。修复：

```bash
docker save "$IMAGE:${TAGS[0]}" -o "$exported_dir/brain-service.docker-image.tar"
```

另外 README 里的 `docker compose up -d` 所需的 compose 文件，不在 docker-deploy 目录里（brain-service 没有 compose 文件），
这一步的文件需要从别处获取。

## 7.4 多架构镜像

### 为什么需要

团队同时有 x86_64 机器（CI、开发机）和 aarch64 机器（Jetson 工作站）。镜像里的二进制是有架构的，
在 Jetson 上运行 x86 镜像会得到：

```
exec /usr/bin/python: exec format error
```

### manifest list（镜像索引）

希望的效果是：**同一个 tag `brain-service:v1.6.2`，x86 机器 pull 到 x86 版本，Jetson pull 到 arm64 版本。**
实现方式是 manifest list（OCI 中称为 image index）：一个 tag 指向一份"清单的清单"，里面按平台列出各自的镜像：

```
brain-service:v1.6.2  (manifest list)
├── linux/amd64 → sha256:aaa...   (= brain-service:v1.6.2_x86_64)
└── linux/arm64 → sha256:bbb...   (= brain-service:v1.6.2_aarch64)
```

`docker pull` 时客户端根据自己的平台自动选择。官方镜像（`python:3.11`、`ubuntu:22.04`）全都是这样发布的，
所以 `FROM python:3.11` 在两种机器上都能用。

### 仓库的做法：分别在原生机器上构建，再合并

CI 的 stages（`bamboo.yaml`）：

```
Image building:
  - X86 Docker building        ← 在 x86_64 agent 上：build_docker.sh --push --tag v1.6.2_x86_64
  - Aarch64 Docker building    ← 在 aarch64 agent 上：build_docker.sh --push --tag v1.6.2_aarch64
Image Merge Tag:
  - Image Merge Tag            ← build_docker.sh --create-manifest --push --tag v1.6.2_<本机架构>
```

合并步骤（`build_docker.sh` 的 `--create-manifest` 分支）：

```bash
full_tag="${TAGS[0]}"                                         # v1.6.2_x86_64
base_tag=$(echo "$full_tag" | sed -E "s/_$(uname -m)$//")     # 去掉本机架构后缀 → v1.6.2
arch_x86_tag="${base_tag}_x86_64"
arch_arm_tag="${base_tag}_aarch64"

docker manifest create "$REG/$IMAGE:$base_tag" \
    "$REG/$IMAGE:$arch_x86_tag" \
    "$REG/$IMAGE:$arch_arm_tag"
docker manifest annotate "$REG/$IMAGE:$base_tag" "$REG/$IMAGE:$arch_x86_tag" --arch amd64
docker manifest annotate "$REG/$IMAGE:$base_tag" "$REG/$IMAGE:$arch_arm_tag" --arch arm64
docker manifest push "$REG/$IMAGE:$base_tag"
```

- `docker manifest create` 只在**仓库层面**组装清单，引用的两个镜像必须已经 push 到仓库，本地不需要有。
- `annotate --arch` 显式标注平台。`create` 一般会从镜像配置里读出平台信息，annotate 多数情况下是保险措施。
- 脚本里检查/开启 `experimental` 是老版本 Docker 的要求，Docker 20.10 之后 `docker manifest` 已不需要实验模式。
- 合并后的 `docker rmi $full_tag || true`：`$full_tag` 是 `v1.6.2_x86_64` 这种不含仓库名的字符串，
  docker 会把它当作镜像名 `v1.6.2_x86_64:latest`，几乎必然不存在，所以这两行实际什么也没删（`|| true` 掩盖了错误）。

### ⚠ 命名约定冲突

- 合并逻辑要求 tag 以 `_x86_64` / `_aarch64` **结尾**（`uname -m` 的输出）。
- `build_docker.sh` 的默认 tag 却是 `amd64_latest` / `arm64_v2025...` —— 架构作**前缀**、而且用的是 Docker 的平台命名（amd64/arm64）。

CI 总是显式传 `--tag v1.6.2_$(uname -m)`，所以 CI 能正常工作；手动在本地不带 `--tag` 构建再试图合并时就会对不上。
一个项目里最好只有一种架构命名约定。

### 另一种做法：buildx 一次构建多平台

```bash
docker buildx create --use --name multi
docker buildx build --platform linux/amd64,linux/arm64 \
    -t robot-harbor.geekplus.com/gino/infra-bin:v1.3.0 --push .
```

buildx 在一台机器上为所有平台构建，并直接推送 manifest list。非本机架构默认通过 **QEMU 用户态模拟**执行 RUN，
对 bazel 编译这种重活，模拟会慢 5～20 倍。所以：

- **重编译型镜像（brain-service）**：仓库现在的"原生机器分别构建 + manifest 合并"是合理选择。
  buildx 也支持把多台原生机器组成一个 builder：`docker buildx create --append --name multi ssh://user@jetson`。
- **轻量镜像（event-uploader、rdp 这种 pip install 为主的）**：buildx + QEMU 一条命令就够了。

⚠ 两种方式不要混用：lab07 实测，用 `docker buildx build --push` 构建的**单平台**镜像，推上去的也是一个 index
（buildx 默认附加 provenance attestation），之后 `docker manifest create` 会报 `... is a manifest list`。
要么加 `--provenance=false`，要么直接用 buildx 多平台构建。仓库的 build.sh 用的是普通 `docker build` + `docker push`，目前不受影响。

### 一个 Dockerfile 支持多架构：`TARGETARCH`

`brain-service/Dockerfile_x86_64` 和 `Dockerfile_aarch64` 唯一的区别是 aarch64 版本的 `bazel build` 多了一个 `-s`。
`universal-uploader` 的两个 Dockerfile 只差 Miniforge 安装包文件名里的架构。维护两份几乎相同的文件，迟早会改了一个忘了另一个。

BuildKit 自动提供一组平台 ARG（在阶段内用 `ARG` 声明后即可使用）：

| ARG | 例子 |
|---|---|
| `TARGETPLATFORM` | `linux/arm64` |
| `TARGETARCH` | `amd64` / `arm64` |
| `BUILDPLATFORM` / `BUILDARCH` | 执行构建的机器的平台 |

合并后的 universal-uploader：

```dockerfile
FROM robot-harbor.geekplus.com/gino/ubuntu:22.04
ARG TARGETARCH
RUN case "$TARGETARCH" in \
        amd64) CONDA_ARCH=x86_64 ;; \
        arm64) CONDA_ARCH=aarch64 ;; \
        *) echo "unsupported arch $TARGETARCH" && exit 1 ;; \
    esac && \
    curl -fsSL -o ~/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${CONDA_ARCH}.sh && ...
```

或者更简单地在 RUN 里用 `$(uname -m)`（原生构建时 `uname -m` 就是目标架构；QEMU 模拟下也会返回目标架构）。

### 查看一个 tag 包含哪些平台

```bash
docker manifest inspect python:3.11 | grep -A3 platform
docker buildx imagetools inspect robot-harbor.geekplus.com/gino/brain-service:v1.6.2
docker image inspect -f '{{.Os}}/{{.Architecture}}' <本地镜像>
```

## 7.5 Docker 镜像 → Singularity/Apptainer：`build_singularity.sh`

HPC 集群上通常不允许运行 Docker daemon（它需要 root），而是用 Singularity（现名 Apptainer）。
Apptainer 可以直接把 Docker 镜像转成单个 `.sif/.simg` 文件，放在 NFS 上供所有计算节点使用：

```bash
DOCKER_SHA="$(docker inspect --format='{{ index .RepoDigests 0 }}' ${FULL_DOCKER_IMAGE_NAME})"
...
singularity build $SIMG_OUTPUT_FILE docker-daemon://$FULL_DOCKER_IMAGE_NAME
ln -s $SIMG_OUTPUT_FILE $OUTPUT_PATH/latest.simg
aws s3 cp $SIMG_OUTPUT_FILE s3://.../$DOCKER_SHA_ONLY.simg
```

- `docker-daemon://` 表示从本地 Docker daemon 读取镜像（另一种是 `docker://` 直接从仓库拉）。
- 用镜像的 **digest**（`RepoDigests`）命名 simg 文件：内容相同的镜像不会重复转换，"已存在就跳过"的逻辑就是基于此。
  注意 `RepoDigests` 只有从仓库 pull 过或 push 过的镜像才有，纯本地 build 的镜像这一项为空，脚本会失败。
- `latest.simg` 软链接 + S3 上的 `latest.simg` 指针文件，和 Docker 的 `latest` tag 是同一个思路。

## 7.6 动手

做 [code/lab07_multiarch](code/lab07_multiarch/)：用 buildx 构建一个双架构镜像推到本地临时仓库，
用 `imagetools inspect` 查看 manifest list，再用 `--platform` 分别运行两个架构的版本。

下一章：[08_review_and_best_practices.md](08_review_and_best_practices.md)
