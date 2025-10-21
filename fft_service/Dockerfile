FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# (Optional) non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health runs on 5000 (ACA probes hit this); no EXPOSE required, but harmless:
EXPOSE 5000

CMD ["python", "fft_service.py"]
