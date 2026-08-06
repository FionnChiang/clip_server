# Stage 1: 前端依赖与初始产物（外网构建时一次性完成）
# node 二进制、node_modules、初始 dist 全部打进运行时镜像：
# 内网启动时可用镜像内 node_modules 离线重建 dist（服务器无需安装 node）
FROM --platform=linux/amd64 node:18 AS frontend-deps
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry=https://registry.npmmirror.com
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend（Web 服务不加载模型，无 torch 依赖）
FROM --platform=linux/amd64 python:3.12-slim

# node 运行时（含 npm/npx），供内网启动时重建前端 dist
COPY --from=frontend-deps /usr/local/bin/node /usr/local/bin/node
COPY --from=frontend-deps /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
COPY --from=frontend-deps /usr/local/bin/npm /usr/local/bin/npm
COPY --from=frontend-deps /usr/local/bin/npx /usr/local/bin/npx

WORKDIR /app

# CJK 字体：OFD 文本渲染依赖（easyofd 使用系统中文字体）
# 换清华 apt 源加速国内构建
RUN (sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true) \
 && (sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true) \
 && apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY server/ /app/server/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# 前端源码 + node_modules + 初始 dist（可被 bind mount 覆盖，启动脚本自动重建 dist）
COPY frontend/ /app/frontend/
COPY --from=frontend-deps /build/frontend/node_modules /app/frontend/node_modules
COPY --from=frontend-deps /build/frontend/dist /app/frontend/dist
# 初始 dist 备份：内网构建失败时回退（挂载目录可能为空）
COPY --from=frontend-deps /build/frontend/dist /app/frontend_dist

RUN mkdir -p /shared/output

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 8000

CMD ["sh", "/app/server/docker-entrypoint.sh"]
