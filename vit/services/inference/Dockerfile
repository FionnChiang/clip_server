FROM --platform=linux/amd64 nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY services/inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

COPY src/ /app/src/
COPY services/inference/app.py /app/app.py

RUN mkdir -p /shared/output

ENV OUTPUT_DIR=/shared/output SRC_DIR=/app

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
