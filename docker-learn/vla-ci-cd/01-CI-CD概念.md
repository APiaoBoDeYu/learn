# 01 CI/CD 是什么

CI/CD 指代码一提交，就由服务器自动完成检查、构建、发布，不用人手工操作。

## CI：持续集成（Continuous Integration）

每次推送代码，自动做代码检查、测试和构建，尽早发现问题。

没有 CI 时，常见的情况是：代码在自己电脑上能跑，合进主分支后才发现别人的机器上编译不过，或者风格检查一堆报错。CI 让每次推送都在同一个干净环境里跑一遍同样的检查。

## CD：有两种含义

| | 持续交付（Continuous Delivery） | 持续部署（Continuous Deployment） |
|---|---|---|
| 自动做到哪一步 | 把产物发布到可以直接部署的地方，比如镜像仓库 | 连部署也自动完成，直接更新生产环境 |
| 上线由谁决定 | 人 | 流水线 |

## 这个项目做到了哪一步

```
代码检查 ──► 构建镜像 ──► 推送到镜像仓库 ──║──► 部署到机器人
└───────── CI ────────┘   └─ 持续交付 ─┘   ║   └── 人工 ──┘
                                     流水线到此结束
```

- **CI**：changelog 检查、ruff 风格检查、SonarQube 扫描、构建镜像。
- **持续交付**：镜像推送到 `robot-harbor.geekplus.com/gino`。
- **没有持续部署**：在机器人上拉取新镜像并重启容器，需要人手动执行 `docker compose up`。

## 相关名词

| 名词 | 在本项目中对应 |
|---|---|
| CI 工具 | Bamboo（Atlassian 的产品，和 Bitbucket 配套） |
| 流水线配置 | `bamboo-specs/bamboo.yaml`，配置写在代码仓库里，叫 "configuration as code" |
| 构建机（agent） | x86_64 机器跑代码检查，Thor 机器 `thor_1` 跑镜像构建 |
| 制品（artifact） | 这里是 Docker 镜像 |
| 制品仓库 | Harbor 镜像仓库 `robot-harbor.geekplus.com` |
