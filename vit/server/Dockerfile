# Stage 1: Build frontend
FROM --platform=linux/amd64 node:18 AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install --registry=https://registry.npmmirror.com
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# CJK 字体：OFD 文本渲染依赖（easyofd 使用系统中文字体）
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY server/ /app/server/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY --from=frontend-build /build/frontend/dist /app/frontend/dist

RUN mkdir -p /shared/output

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
