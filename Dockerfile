FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ML_GRPC_LISTEN=0.0.0.0:50051 \
    MODEL_DIR=/models \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY ml_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && useradd --uid 10001 --create-home app

COPY --chown=app:app ml_service /app/ml_service
USER app
EXPOSE 50051
CMD ["python", "-m", "ml_service.server"]
