# 纯运行时镜像：仅包含 OS、Python、PyTorch 及 Python 依赖。
# 项目代码 (vit/) 和模型文件 (models/) 通过 volume 挂载进入容器。
# 这样修改代码或配置后无需重新构建镜像。
#
# 注意：依赖版本与 services/training 保持一致（cu124 + 锁版本 + cuDNN），
# 保证离线环境行为一致。

FROM --platform=linux/amd64 nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN (sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true) \
 && (sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true) \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

COPY vit/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    && rm /tmp/requirements.txt

RUN mkdir -p /workspace/data /workspace/output /workspace/models

ENV PYTHONPATH=/workspace:$PYTHONPATH

CMD ["python", "scripts/verify.py"]
