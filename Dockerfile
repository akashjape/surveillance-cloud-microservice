FROM python:3.10-slim

WORKDIR /app

# Install System Dependencies for OpenCV, OpenVINO & Git LFS
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hydrate any Git LFS pointer files at build time
RUN git config --global --add safe.directory '*' && \
    git lfs install && \
    (git lfs pull || true) && \
    python3 download_models.py

ENV PORT=8080
EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
