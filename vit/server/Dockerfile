FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server/ /app/server/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

RUN mkdir -p /shared/output

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
