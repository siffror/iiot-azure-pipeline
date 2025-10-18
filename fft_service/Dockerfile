# Dockerfile (fft-service)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fft_service.py .
# ingen exponering behövs för ACA, lyssnar på 5000 för health
CMD ["python", "fft_service.py"]
