ARG BASE=python:3.11-slim
FROM ${BASE}
WORKDIR /app
COPY app.py entrypoint_noexec.sh entrypoint_exec.sh ./
RUN chmod +x /app/entrypoint_*.sh
ENV PYTHONUNBUFFERED=1
# 默认用"没有 exec"的版本，各个用例通过 --entrypoint 切换
ENTRYPOINT ["/app/entrypoint_noexec.sh"]
