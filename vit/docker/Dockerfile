# 纯运行时镜像：仅包含 OS、Python、PyTorch 及 Python 依赖。
# 项目代码 (vit/) 和模型文件 (models/) 通过 volume 挂载进入容器。
# 这样修改代码或配置后无需重新构建镜像。
#
# 如果 CUDA 13.0 镜像不可用，构建时传入 --build-arg CUDA_TAG=12.8.0-cudnn-runtime-ubuntu22.04

ARG CUDA_TAG=13.0.0-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

COPY vit/requirements.txt /tmp/requirements.txt

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN mkdir -p /workspace/data /workspace/output /workspace/models

ENV PYTHONPATH=/workspace:$PYTHONPATH

CMD ["python", "scripts/verify.py"]
