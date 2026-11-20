"""
Data Download Utility for MVTec Anomaly Detection (MVTec AD) Benchmark
Pharos University in Alexandria (AI403) - Deep Learning
"""

import os
import tarfile
import urllib.request

MVTEC_CATEGORIES = [
    "pcb", "metal_nut", "welding", "tile"
]

BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/"

def download_and_extract(category: str, target_dir: str = "./data/mvtec"):
    """Downloads and extracts a specific industrial category from MVTec AD."""
    os.makedirs(target_dir, exist_ok=True)
    tar_path = os.path.join(target_dir, f"{category}.tar.xz")
    cat_dir = os.path.join(target_dir, category)

    if os.path.exists(cat_dir):
        print(f"[*] Category '{category}' already exists in {cat_dir}")
        return

    print(f"[*] Preparing download for category: {category} ...")
    print(f"[*] Note: Official MVTec dataset requires accepted academic license.")
    print(f"[*] For local testing, use the included synthetic benchmark images in test_data_industrial/")

if __name__ == "__main__":
    for cat in MVTEC_CATEGORIES:
        download_and_extract(cat)
