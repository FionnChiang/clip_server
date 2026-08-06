#!/bin/sh
# 容器启动入口：
# 1. 用镜像内 node_modules 离线重建前端 dist（服务器无需安装 node）
# 2. 构建失败时回退到镜像内置的初始 dist，保证服务可用
set -e

echo "[entrypoint] Rebuilding frontend dist (offline)..."
if (cd /app/frontend && npm run build) 2>/tmp/frontend-build.log; then
    echo "[entrypoint] Frontend build OK"
else
    echo "[entrypoint] Frontend build FAILED, falling back to baked-in dist (see /tmp/frontend-build.log)"
    if [ ! -f /app/frontend/dist/index.html ]; then
        cp -r /app/frontend_dist/. /app/frontend/dist/ 2>/dev/null || true
    fi
fi

exec uvicorn server.main:app --host 0.0.0.0 --port 8000
