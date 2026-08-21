FROM python:3.10-slim

WORKDIR /app

# Install Modern OpenCV System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port $PORT