# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=prod \
    LOG_DIR=/app/data/logs \
    METRICS_DIR=/app/data/metrics \
    REFERENCE_DIR=/app/data/reference
WORKDIR /app
COPY --from=builder /install /usr/local
COPY api/ api/
COPY artifacts/ artifacts/
COPY model_registry/ model_registry/
COPY data/reference/ data/reference/
COPY requirements.txt ./
EXPOSE 8080
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
