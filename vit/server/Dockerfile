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
