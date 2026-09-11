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
    python3 -c "\
import os, urllib.request; \
models = ['text_encoder.onnx', 'image_encoder.onnx', 'pa100k_osnet.bin', 'dinov2_vits14.bin', 'arcface_resnet50_survface.bin', 'osnet_17_15.bin']; \
for m in models: \
    if os.path.exists(m) and os.path.getsize(m) < 1000: \
        url = f'https://media.githubusercontent.com/media/akashjape/surveillance-cloud-microservice/main/{m}'; \
        print(f'Downloading full {m} from {url}...'); \
        urllib.request.urlretrieve(url, m); \
        print(f'Downloaded {m}: {os.path.getsize(m)} bytes') \
"

ENV PORT=8080
EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
