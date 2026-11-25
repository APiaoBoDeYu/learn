ARG BASE=python:3.11-slim
FROM ${BASE}

# ARG：只在构建期存在
ARG BUILD_ONLY=i-am-only-visible-during-build
# ARG 接收 + ENV 固化：运行期也能看到
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="lab01" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /srv/app

# 构建期能看到两个变量
RUN echo "build time: BUILD_ONLY=${BUILD_ONLY} APP_VERSION=${APP_VERSION}" > /build-info.txt

COPY app.py .

EXPOSE 8000
CMD ["python", "app.py"]
