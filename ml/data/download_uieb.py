"""
download_uieb.py — Download the UIEB dataset via the Kaggle API.

UIEB (Underwater Image Enhancement Benchmark):
  890 paired raw / reference images + 60 challenge images.
  We use only the 890 paired images for supervised training.

Setup (do this once):
  1. Create a free account at https://www.kaggle.com
  2. Account → Settings → API → Create New Token → kaggle.json downloads
  3. Place kaggle.json at:
       Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
       Linux   : ~/.kaggle/kaggle.json
  4. pip install kaggle

Usage:
  python ml/data/download_uieb.py
"""

import os
import sys
from pathlib import Path

# ── Target directories ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent          # ml/data/
RAW_DIR  = DATA_DIR / "raw-890"
REF_DIR  = DATA_DIR / "reference-890"

# ── Kaggle dataset identifier ─────────────────────────────────────────────────
# Find the correct ID in the dataset URL:
# https://www.kaggle.com/datasets/<OWNER>/<DATASET-NAME>
KAGGLE_DATASET = "larjeck/uieb-dataset-raw-and-reference"


def download() -> None:
    # Skip if already downloaded
    if RAW_DIR.exists() and REF_DIR.exists():
        raw_n = len([f for f in RAW_DIR.iterdir() if f.is_file()])
        ref_n = len([f for f in REF_DIR.iterdir() if f.is_file()])
        print(f"Dataset already present — {raw_n} raw, {ref_n} reference images.")
        return

    print(f"Downloading UIEB from Kaggle ({KAGGLE_DATASET}) …")

    exit_code = os.system(
        f'kaggle datasets download -d {KAGGLE_DATASET} -p "{DATA_DIR}" --unzip'
    )

    if exit_code != 0:
        print(
            "\n[ERROR] Kaggle download failed. Checklist:\n"
            "  1. pip install kaggle\n"
            "  2. kaggle.json placed at C:\\Users\\<You>\\.kaggle\\kaggle.json\n"
            "  3. File permissions: chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac)\n"
            f"  4. Dataset ID is correct: {KAGGLE_DATASET}\n"
            "     (verify at https://www.kaggle.com/datasets)\n"
        )
        sys.exit(1)

    # Verify expected folder structure
    missing = [d for d in (RAW_DIR, REF_DIR) if not d.exists()]
    if missing:
        print(
            f"\n[ERROR] Expected folders not found after download:\n"
            + "\n".join(f"  {d}" for d in missing)
            + "\n\nThe Kaggle dataset folder structure may differ from expected.\n"
            f"Please check the downloaded contents of: {DATA_DIR}\n"
            f"Then update RAW_DIR / REF_DIR constants in {__file__}"
        )
        sys.exit(1)

    raw_n = len([f for f in RAW_DIR.iterdir() if f.is_file()])
    ref_n = len([f for f in REF_DIR.iterdir() if f.is_file()])
    print(f"\nDownload complete — {raw_n} raw, {ref_n} reference images.")


if __name__ == "__main__":
    download()
