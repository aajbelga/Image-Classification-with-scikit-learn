"""
download_data.py — Download and extract the CIFAR-10 dataset.
Run this once before training.
"""

import urllib.request
import tarfile
import os

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEST = "data/cifar-10-python.tar.gz"

def download():
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/cifar-10-batches-py"):
        print("✅ Dataset already exists at data/cifar-10-batches-py")
        return

    print(f"⬇️  Downloading CIFAR-10 (~163 MB) ...")
    urllib.request.urlretrieve(URL, DEST, reporthook=_progress)
    print("\n📦 Extracting ...")
    with tarfile.open(DEST, "r:gz") as tar:
        tar.extractall("data/")
    os.remove(DEST)
    print("✅ Done! Dataset at data/cifar-10-batches-py")

def _progress(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size)
    print(f"\r   {min(pct, 100)}%", end="", flush=True)

if __name__ == "__main__":
    download()
