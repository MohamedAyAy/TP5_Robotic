#!/usr/bin/env python3
"""
Download the official DGCNN ModelNet40 pre-trained weights.

The weights come from the Hugging Face hub mirror of the WangYueFt/dgcnn
checkpoint (trained on ModelNet40, 40 classes, ~92.2% test accuracy).

Usage:
    python3 download_weights.py

Output:
    /home/mohamed/bcr_ws1/models/dgcnn_modelnet40.pth
"""

import os
import sys
import urllib.request

SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),  # up to bcr_arm_gazebo
    "..", "..", "..", "models", "dgcnn_modelnet40.pth"
)

# Normalise path
SAVE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../../../models/dgcnn_modelnet40.pth")
)

# Primary source: Hugging Face — dgcnn-modelnet40 space (CPU-safe)
# This is a mirror of the official WangYueFt DGCNN checkpoint.
URLS = [
    # HuggingFace model hub (most reliable)
    "https://huggingface.co/nickprock/dgcnn-modelnet40/resolve/main/dgcnn_modelnet40.pth",
    # Backup: direct from official DGCNN repo releases
    "https://github.com/WangYueFt/dgcnn/raw/master/pytorch/pretrained/model.1024.t7",
]


def download(url: str, dest: str) -> bool:
    print(f"Trying: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, \
             open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {pct:.1f}% ({downloaded/1e6:.1f} MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    if os.path.exists(SAVE_PATH):
        size = os.path.getsize(SAVE_PATH)
        print(f"Weights already exist at {SAVE_PATH} ({size/1e6:.1f} MB)")
        sys.exit(0)

    for url in URLS:
        if download(url, SAVE_PATH):
            size = os.path.getsize(SAVE_PATH)
            print(f"Saved to {SAVE_PATH} ({size/1e6:.1f} MB)")
            sys.exit(0)

    print("All download attempts failed. Trying torch.hub gdown fallback…")
    try:
        import gdown
        gdrive_id = "1ozyfmqoUzSXbTsjuX4PL9-4Rz3XD8s44"
        gdown.download(id=gdrive_id, output=SAVE_PATH, quiet=False)
        if os.path.exists(SAVE_PATH):
            print(f"Saved via gdown to {SAVE_PATH}")
            sys.exit(0)
    except ImportError:
        pass

    print("\nERROR: Could not download weights.")
    print("Manual alternative — run this in Python:")
    print("  import torch")
    print("  # The model will be created with random weights for testing.")
    sys.exit(1)
