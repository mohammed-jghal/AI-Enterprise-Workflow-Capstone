FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80
WORKDIR /app

# Runtime dependencies only (see requirements-dev.txt for test/notebook deps).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY . .
# Bake the trained model into the image: the artifact (models/*.pkl) is
# gitignored, so the image must train it. This runs the full SARIMA
# training (7 series x 3 models) and takes several minutes; it is the
# slowest build step by design, not a hang. Ingestion regenerates the
# committed data/output products from data/input.
RUN python run_ingest.py && python run_train.py \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:80/health', timeout=3)"
CMD ["python", "./run_app.py"]
