#!/usr/bin/env python3
"""Build-time helper to ensure all model weights are hydrated (not Git LFS pointers)."""
import os
import urllib.request

MODELS = [
    "text_encoder.onnx",
    "image_encoder.onnx",
    "pa100k_osnet.bin",
    "dinov2_vits14.bin",
    "arcface_resnet50_survface.bin",
    "osnet_17_15.bin",
]

def main():
    print("Checking model files on disk...")
    for m in MODELS:
        if os.path.exists(m) and os.path.getsize(m) < 1000:
            url = f"https://media.githubusercontent.com/media/akashjape/surveillance-cloud-microservice/main/{m}"
            print(f"Downloading full binary for {m} from {url}...")
            urllib.request.urlretrieve(url, m)
            print(f"Successfully downloaded {m} ({os.path.getsize(m)} bytes)")
        elif os.path.exists(m):
            print(f"Model {m} verified present ({os.path.getsize(m)} bytes)")
        else:
            url = f"https://media.githubusercontent.com/media/akashjape/surveillance-cloud-microservice/main/{m}"
            print(f"File {m} missing. Downloading from {url}...")
            try:
                urllib.request.urlretrieve(url, m)
                print(f"Successfully downloaded {m} ({os.path.getsize(m)} bytes)")
            except Exception as e:
                print(f"Warning: could not download {m}: {e}")

    print("Model hydration check completed.")

if __name__ == "__main__":
    main()
