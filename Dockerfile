FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ml_service /app/ml_service

WORKDIR /app/ml_service

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
