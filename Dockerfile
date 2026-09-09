FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY api.py .
COPY config.py .
COPY database ./database
COPY rag ./rag
COPY services ./services
COPY tools ./tools
COPY bot ./bot
COPY utils ./utils

CMD ["python", "main.py"]
