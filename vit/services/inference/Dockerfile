# 推理服务镜像
# 说明：
# - 基础镜像带 cuDNN（torch cu124 wheel 运行时需要 libcudnn.so.9）
# - torch/torchvision/transformers 锁定版本，保证离线环境可复现
# - CLIP 模型权重直接 COPY 进镜像（/shared/models）：checkpoint 中
#   config.model.path 指向 /shared/models，推理时直接命中
FROM --platform=linux/amd64 nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN (sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true) \
 && (sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true) \
 && apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY services/inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

COPY src/ /app/src/
COPY services/inference/app.py /app/app.py
COPY models/ /shared/models/

RUN mkdir -p /shared/output

ENV OUTPUT_DIR=/shared/output SRC_DIR=/app

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
