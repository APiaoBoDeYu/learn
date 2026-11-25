#!/bin/bash
# 仿照 universal-uploader/deploy.sh：首次运行时生成 secret 文件，已存在则不覆盖。
# 注意这里故意没有 pipefail：`tr < /dev/urandom | head` 中 tr 会收到 SIGPIPE，
# 开了 pipefail 整条管道会被判定为失败。
set -eu
cd "$(dirname "$0")"
mkdir -p secrets
[[ -f secrets/pg_user ]]     || echo "labuser" > secrets/pg_user
[[ -f secrets/pg_password ]] || tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 16 > secrets/pg_password
chmod 644 secrets/pg_user secrets/pg_password
echo "secrets ready:"; ls -l secrets
