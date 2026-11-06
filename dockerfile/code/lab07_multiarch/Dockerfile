# syntax=docker/dockerfile:1
# 一个 Dockerfile 支持多架构：用 BuildKit 自动注入的平台 ARG
ARG BASE=debian:bookworm-slim
FROM ${BASE}

ARG TARGETPLATFORM
ARG TARGETARCH
ARG BUILDPLATFORM

# 演示"按架构选择下载包名"（同 universal-uploader 的 Miniforge、local_clipping 的 ffmpeg）
RUN case "$TARGETARCH" in \
        amd64) PKG_ARCH=x86_64 ;; \
        arm64) PKG_ARCH=aarch64 ;; \
        *) echo "unsupported arch: $TARGETARCH" >&2; exit 1 ;; \
    esac && \
    echo "built on ${BUILDPLATFORM} for ${TARGETPLATFORM}; would download Miniforge3-Linux-${PKG_ARCH}.sh" > /build-info.txt

CMD ["sh", "-c", "cat /build-info.txt; echo \"runtime uname -m: $(uname -m)\""]
