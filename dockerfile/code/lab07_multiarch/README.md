# lab07：多架构镜像与 manifest list

对应第 7 章。

## A. buildx 一条命令构建双架构（`run.sh`）

```bash
cd lab07_multiarch
./run.sh
```

脚本做的事：起一个本地 `registry:2` → 装 QEMU binfmt → 建 docker-container 驱动的 builder →
`buildx build --platform linux/amd64,linux/arm64 --push` → `imagetools inspect` → 分别 `--platform` 运行。

**实测**（在 Jetson Thor 上构建，amd64 那份经 QEMU 模拟）：

```
== imagetools inspect
MediaType: application/vnd.oci.image.index.v1+json
Manifests:
  Platform:    linux/amd64
  Platform:    linux/arm64
  Platform:    unknown/unknown     vnd.docker.reference.type: attestation-manifest
  Platform:    unknown/unknown     vnd.docker.reference.type: attestation-manifest
== docker run --platform linux/amd64
built on linux/arm64 for linux/amd64; would download Miniforge3-Linux-x86_64.sh
runtime uname -m: x86_64
== docker run --platform linux/arm64
built on linux/arm64 for linux/arm64; would download Miniforge3-Linux-aarch64.sh
runtime uname -m: aarch64
```

- 一个 tag 下有两个平台的镜像，另外两个 `unknown/unknown` 是 buildx 默认附带的 provenance attestation（构建来源证明），各自指向一个平台镜像。
- `TARGETARCH` 正确地选出了不同的包名 —— 这就是合并 `Dockerfile_x86_64` / `Dockerfile_aarch64` 的方法。
- 在 QEMU 模拟下，`uname -m` 返回的是**目标架构** x86_64。

## B. 仓库的做法：分别构建 + `docker manifest create`

模拟 CI 的 "X86 Docker building / Aarch64 Docker building / Image Merge Tag" 三个任务：

```bash
REG=localhost:5000
# 单平台构建两次，分别打上 _x86_64 / _aarch64 后缀的 tag（这里用 buildx 模拟"两台机器"）
docker buildx build --platform linux/amd64 -t $REG/lab07:v2_x86_64  --push .
docker buildx build --platform linux/arm64 -t $REG/lab07:v2_aarch64 --push .

# 合并（同 build_docker.sh --create-manifest；本地 http 仓库需要 --insecure）
docker manifest create --insecure $REG/lab07:v2 $REG/lab07:v2_x86_64 $REG/lab07:v2_aarch64
docker manifest annotate $REG/lab07:v2 $REG/lab07:v2_x86_64  --arch amd64
docker manifest annotate $REG/lab07:v2 $REG/lab07:v2_aarch64 --arch arm64
docker manifest push --insecure $REG/lab07:v2

docker manifest inspect --insecure $REG/lab07:v2
```

**实测：上面这样直接做会失败。**

```
$ docker buildx imagetools inspect $REG/lab07:v2_x86_64 | grep MediaType
MediaType: application/vnd.oci.image.index.v1+json        ← 单平台构建，推上去的却是一个 index
$ docker manifest create --insecure $REG/lab07:v2 ...
localhost:5000/lab07:v2_x86_64 is a manifest list
```

原因：buildx 默认给镜像附加 provenance attestation，所以即使只构建一个平台，推上去的也是"镜像 + attestation"组成的 index。
`docker manifest create` 只接受单个镜像 manifest。给两次 buildx build 加上 `--provenance=false`：

```bash
docker buildx build --provenance=false --platform linux/amd64 -t $REG/lab07:v3_x86_64  --push .
docker buildx build --provenance=false --platform linux/arm64 -t $REG/lab07:v3_aarch64 --push .
docker manifest create --insecure $REG/lab07:v3 $REG/lab07:v3_x86_64 $REG/lab07:v3_aarch64
docker manifest annotate $REG/lab07:v3 $REG/lab07:v3_x86_64  --arch amd64
docker manifest annotate $REG/lab07:v3 $REG/lab07:v3_aarch64 --arch arm64
docker manifest push --insecure $REG/lab07:v3
```

实测：`imagetools inspect v3_x86_64` 的 MediaType 变为 `application/vnd.oci.image.manifest.v1+json`，
`Created manifest list localhost:5000/lab07:v3`，push 成功，`manifest inspect` 列出 `arm64` / `amd64` 两个平台。

**这对仓库的意义**：CI 现在用的是 `docker build` + `docker push`（默认 docker driver，不附加 attestation），所以 `--create-manifest` 能工作。
如果将来有人把 build.sh 改成 `docker buildx build --push`，manifest 合并这一步就会以 `is a manifest list` 失败。

## C. 架构不匹配会怎样

```bash
docker run --rm --platform linux/amd64 $REG/lab07:v3_x86_64      # 实测：runtime uname -m: x86_64（经 QEMU）
docker run --rm --privileged tonistiigi/binfmt --uninstall qemu-x86_64   # 卸载 x86 模拟器（Jetson 上）
docker run --rm --platform linux/amd64 $REG/lab07:v3_x86_64      # 实测：exec /usr/bin/sh: exec format error
docker run --privileged --rm tonistiigi/binfmt --install all            # 装回来
```

这就是 `local_clipping/Dockerfile` 在 aarch64 上下载 amd64 版 ffmpeg 会遇到的错误。

## 清理

```bash
docker rm -f lab07-registry
docker buildx rm lab07-builder
# run.sh 往内核 binfmt_misc 注册了所有 QEMU 模拟器（重启前一直有效）。想恢复原状：
docker run --rm --privileged tonistiigi/binfmt --uninstall "qemu-*"
ls /proc/sys/fs/binfmt_misc/
```
