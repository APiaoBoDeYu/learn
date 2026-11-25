# 02 bamboo.yaml：流水线定义

文件：`embodied_algo/bamboo-specs/bamboo.yaml`

Bitbucket 收到推送后通知 Bamboo，Bamboo 读取这个文件，按里面的定义执行。

## 先打个比方：一家自动化工厂

把 Bamboo 想成一家工厂，代码仓库是原料仓库。每次 `git push` 就送来一批新原料，工厂按一张"生产工单"加工，`bamboo.yaml` 就是这张工单。

| 工厂里 | Bamboo 里 | 说明 |
|---|---|---|
| 生产线 | Plan | 这里叫 `embodied_algo` |
| 车间 | Stage | 按顺序走，前一个车间不合格，后面的不开工 |
| 工位 | Job | 同一车间的工位可以并行，每个工位分到一台构建机（agent），在一个干净的容器里干活 |
| 工序 | Task | 工位上的具体步骤，比如拉代码、跑脚本 |
| 工位对机器的要求 | requirements | 比如必须是 x86_64，或必须是 Thor 那台 arm64 机器 |
| 每个分支一条复制的生产线 | Plan Branch | `feature/xxx` 和 `dev` 各跑各的，互不影响 |

配置写在仓库里（Bamboo Specs），Bamboo 会自动读取 `bamboo-specs/bamboo.yaml` 并更新 Plan。改这个文件就是改 CI，不需要去网页上点。

## 整体结构

```yaml
version: 2
plan:                          # 这条流水线叫什么
  project-key: HIG
  key: HIEMBODIEDALGO          # plan key 不能含下划线
  name: embodied_algo

variables:                     # 所有任务共享的环境变量
  OMP_CANCELLATION: "true"
  CI: "true"

stages:                        # 有哪些阶段，按顺序执行
  - Code inspection:
      jobs: [Style validation]
  - Code scanning:
      jobs: [SonarQube scanning]
  - Image building:
      jobs: [Aarch64 VLA image building]

Style validation:              # 每个 job 的具体定义
  requirements: ...            # 跑在哪类机器上
  docker: ...                  # 在哪个容器里跑
  tasks: ...                   # 依次执行哪些步骤
```

```
push 代码
   │
   ▼
[Code inspection]  → Style validation             (x86_64)
   │ 通过
   ▼
[Code scanning]    → SonarQube scanning           (x86_64)
   │ 通过
   ▼
[Image building]   → Aarch64 VLA image building   (thor_1, arm64)
```

三个层级：

- **stage（阶段）**：按顺序执行。前一个阶段失败，后面的不再执行。
- **job（任务）**：同一阶段里的多个 job 可以并行。这里每个阶段只有一个 job。
- **task（步骤）**：一个 job 里按顺序执行的步骤，比如 `checkout` 拉代码、`script` 跑脚本。

每个 job 的 `checkout` 都带 `force-clean-build: true`，每次清空工作目录重新拉代码，避免上次构建留下的文件干扰。

## script 任务是什么

`script:` 是 Bamboo 的一种 task 类型，意思是"在构建机上执行下面这段脚本"。job 配了 `docker:`，脚本就跑在那个 Linux 容器里。内容就是普通的 Linux shell 脚本。

### Bamboo 怎么执行它

1. 把文本里所有 `${bamboo.xxx}` 替换成实际值。
2. 把替换后的内容写进一个临时 `.sh` 文件。
3. 在 Linux 上执行这个文件，和你在终端里跑 `./xxx.sh` 一样。

注意 `${bamboo.xxx}` 和 shell 的 `$VAR` 不是一回事：前者在脚本**运行前**由 Bamboo 做文本替换，后者在运行时由 bash 解析。

常用的 Bamboo 变量：

| 变量 | 含义 |
|---|---|
| `${bamboo.build.working.directory}` | 代码拉下来的目录 |
| `${bamboo.planRepository.branchName}` | 当前构建的分支名 |
| `${bamboo.buildNumber}` | 本次构建编号 |
| `${bamboo.GEEK_DOCKER_KEY}` 等 | 在 Bamboo 界面里配置的密钥，不写在代码里，日志里会被打码 |

### YAML 的 `|`：原样保留多行文本

```yaml
- script: |
    #!/bin/bash
    第一行
    第二行
```

`|` 是 YAML 语法，表示下面缩进的所有行原样当作一个多行字符串，换行都保留。YAML 只负责装这段文本，不理解里面的内容。

`script:` 后面也可以跟一个列表（SonarQube 那个 job 就是这么写的），每一项是一条命令。

### 第一行 `#!/bin/bash`

这行叫 shebang，告诉系统用 bash 解释这个文件。Image building 的脚本写了它。Style validation 的脚本没写，这种情况下一般用默认的 `/bin/sh` 执行。脚本里如果用了 `[[ ]]`、数组这类 bash 专有语法，就要加上 shebang。

## 阶段 1：Code inspection（x86_64）

在 `brain-service-dev:v1.2.0` 容器里执行。

**changelog 处理**，按分支区分：

| 分支 | 做什么 |
|---|---|
| `feature/*` `hotfix/*` | 用 `validate_changelog.sh --strict` 检查分支里必须有合法的 changelog 片段文件，没有就构建失败 |
| `dev` `master` `release/*` | 先校验片段，再把 `changelog.d/` 下的片段合并进 `release_notes.md`，删掉片段，由 CI Bot 自动提交并推送回该分支 |
| 其他 | 跳过 |

这样设计是为了避免冲突：每人在功能分支上各写一个独立的片段文件，合并时不会在 `release_notes.md` 上打架，最后由 CI 统一合并。

两个细节：

- **防止无限循环**：CI Bot 自己的提交也会触发一次新的构建。脚本检查最后一条提交信息是否包含 `[ci] merge changelog fragments`，是的话跳过合并。
- **临时换成 HTTPS 推送**：仓库 remote 是 SSH 地址，容器里没有 SSH 密钥。脚本把 Bamboo 变量里的 token（`BRAIN_ACCESS_TOKEN`）做 URL 编码后拼进 HTTPS 地址，push 完再把 remote 改回 SSH。

**取消过期构建**：`!include 'cancel_stale_builds.yaml'` 引入另一个文件里的步骤。对 `dev`、`master`、`release/*` 以外的分支，如果短时间内连续 push 了几次，它会通过 Bamboo REST API 找出同一分支上更早的、还在排队或运行中的构建并停掉，只保留最新这次，节省构建机。这个步骤出任何错都只打 WARN，脚本最后强制 `exit 0`，不会让构建失败。

**风格检查**：分别在 `brain/vla/openpi` 和 `body/gino` 目录里跑 `ruff check`，这样每个子项目用自己的 `pyproject.toml`。末尾的 `|| true` 让它即使报错也不让构建失败，只给出提示。

## 阶段 2：Code scanning（x86_64）

SonarQube 是代码质量和安全扫描平台，能找出潜在 bug、重复代码和安全漏洞，结果显示在它的网页上。

1. 运行 `install_sonar_scanner.sh` 安装扫描器。
2. 执行 `sonar-scanner`，按分支名上报，项目名 `embodied_algo`。
   - `-Dsonar.exclusions=**/third_party/**`：不扫第三方代码。
   - `-Dsonar.c.file.suffixes=-` 等：把 C/C++/Objective-C 的后缀设成 `-`，等于关掉这几种语言的扫描。
   - `SONAR_HOST_URL`、`SONAR_LOGIN` 来自 Bamboo 变量。

注释里说这条命令必须写成一行，因为 Bamboo 解析多行命令有问题。

## 阶段 3：Aarch64 VLA image building（thor_1）

这是和镜像有关的阶段，也是 CI 和 `docker-deploy/` 的连接点。它构建并推送 `vla-brain` 和 `vla-body` 两个 release 镜像。基础镜像 `vla-brain:1.0.0`、`vla-body:1.0.0` 已经在 Harbor 里，CI 从不重建，只在上面叠加业务代码。

### 运行环境

```yaml
requirements:
  - system: thor_1                         # 只在 Thor 构建机上跑，镜像只能在 arm64 上构建
docker:
  image: robot-harbor.geekplus.com/gino/brain-service:v1.0.0_thor
  volumes:
    "/var/run/docker.sock": "/var/run/docker.sock"
  docker-run-arguments:
    - "--net=host"
    - "--entrypoint=/usr/bin/tail"
```

- 任务本身在一个容器里执行。
- 挂载 `/var/run/docker.sock` 后，容器里的 `docker` 命令实际连接的是 Thor 宿主机上的 Docker 守护进程，`docker build` 也由宿主机执行。这种做法叫 Docker-out-of-Docker（DooD），比在容器里再装一个完整 Docker（Docker-in-Docker）简单。
- `--entrypoint=/usr/bin/tail` 让容器保持运行，Bamboo 再通过 `docker exec` 往里面送命令。这个镜像原本的 ENTRYPOINT 会启动 brain-service，这里不需要。
- `--net=host`：容器直接使用宿主机网络。

### 执行的脚本，逐段讲解

```bash
#!/bin/bash
set -euo pipefail
```

安全开关，CI 脚本里很常见：

| 选项 | 作用 |
|---|---|
| `-e` | 任何命令失败就立即退出 |
| `-u` | 用到未定义的变量就报错 |
| `-o pipefail` | 管道 `a \| b` 里任何一个命令失败，整条管道都算失败 |

```bash
WD="${bamboo.build.working.directory}"
```

把代码目录存进 shell 变量 `WD`。`${bamboo...}` 已被 Bamboo 提前替换，`WD` 是普通 bash 变量。

```bash
DECISION="$(bash "${WD}/docker-deploy/vla/should_build.sh" "$WD" "${bamboo.planRepository.branchName}")"
echo "$DECISION"
```

`$(...)` 执行里面的命令，把输出存进变量。`should_build.sh` 输出两行：

```
VLA_RELEASE_SHOULD_BUILD=true
VLA_RELEASE_TAG=test-feature-haiyang.yu-dev-7c7a95b1
```

`echo` 把结果打印到构建日志，方便排查。

```bash
get() { echo "$DECISION" | sed -n "s/^$1=//p"; }
RELEASE_SHOULD_BUILD="$(get VLA_RELEASE_SHOULD_BUILD)"
RELEASE_TAG="$(get VLA_RELEASE_TAG)"
```

定义一个小函数 `get`，按键名取值。`sed -n "s/^KEY=//p"` 的意思是：找到以 `KEY=` 开头的行，删掉 `KEY=`，打印剩下的部分。按键名而不是按行号取值，以后 `should_build.sh` 多输出几行也不受影响。

```bash
if [ "$RELEASE_SHOULD_BUILD" != "true" ]; then
  echo "NO VLA IMAGE BUILD TRIGGERED."
  exit 0
fi
```

不需要构建就提前退出。`exit 0` 表示成功，Bamboo 上显示绿色。

```bash
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com -o get-docker.sh
  sh get-docker.sh
fi
```

`command -v docker` 检查有没有 docker 命令，`>/dev/null 2>&1` 丢掉输出，只看结果。没有就下载 Docker 官方安装脚本并执行。容器里只需要命令行工具，守护进程用的是宿主机的。

```bash
export GEEK_DOCKER_USER="${bamboo.GEEK_DOCKER_USER}"
export GEEK_DOCKER_KEY="${bamboo.GEEK_DOCKER_KEY}"
```

`export` 把变量变成环境变量，后面调用的 `build.sh` 子进程才能读到 Harbor 的账号密码。不加 `export`，变量只在当前脚本里可见。

```bash
"${WD}/docker-deploy/vla/build.sh" --tag "$RELEASE_TAG" --push
echo "VLA RELEASE IMAGE BUILD FINISHED: ${RELEASE_TAG}"
```

调用真正的构建脚本，构建并推送镜像。因为开了 `set -e`，`build.sh` 失败时脚本在这里停下，最后的 echo 不会执行，Bamboo 把这次构建标成失败。

### 本地验证

除了 `${bamboo.xxx}`，脚本的其他部分都能在本地终端直接跑。想知道自己的分支会不会触发构建，在仓库根目录执行：

```bash
bash docker-deploy/vla/should_build.sh "$PWD" feature/haiyang.yu/dev
```

`should_build.sh` 和 `build.sh` 的细节见下一章。

## 新手容易踩的点

- **feature 分支没写 changelog 片段，第一个阶段就会失败**，后面的扫描和镜像构建都不会执行。
- **只有 ruff 和取消过期构建是"失败也不拦"的**。changelog 校验、Sonar 扫描和镜像构建失败都会让构建变红。
- **非 release、非 master 分支默认不构建镜像**。想在功能分支上试构建，提交标题里要带 `[build][vla]`。
