"""
dataset.py — Image-level train/val/test split + tf.data patch pipeline.

WHY image-level split (not patch-level)?
-----------------------------------------
If we split patches directly, patches from the SAME underwater photo can
appear in both the training and validation sets.  The model then "sees" its
own validation data during training, inflating val scores artificially.
Splitting at the IMAGE level first guarantees each photo lands in exactly
one split, preventing this data-leakage.

Pipeline flow
-------------
  image paths  →  flat_map (load + extract patches)  →  shuffle  →  batch
"""

import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

# patches.py lives next to this file
sys.path.insert(0, str(Path(__file__).parent))
from patches import PATCH_SIZE, extract_patches

# ── Default data directories (override when running on Colab/Kaggle) ──────────
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR  = DATA_DIR / "raw-890"
REF_DIR  = DATA_DIR / "reference-890"

SEED      = 42
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── 1. Collect matched image pairs ────────────────────────────────────────────
def get_pairs(
    raw_dir: Path = RAW_DIR,
    ref_dir: Path = REF_DIR,
) -> List[Tuple[str, str]]:
    """
    Scan raw_dir and ref_dir for images with matching filenames (stem).
    Returns a sorted list of (raw_path, ref_path) string tuples.
    """
    raw = {p.stem: p for p in raw_dir.glob("*") if p.suffix.lower() in _IMG_EXTS}
    ref = {p.stem: p for p in ref_dir.glob("*") if p.suffix.lower() in _IMG_EXTS}

    common = sorted(raw.keys() & ref.keys())
    if not common:
        raise FileNotFoundError(
            f"No matching image pairs found.\n"
            f"  raw dir : {raw_dir}\n"
            f"  ref dir : {ref_dir}\n"
            "Run:  python ml/data/download_uieb.py"
        )

    print(f"Found {len(common)} paired images.")
    return [(str(raw[k]), str(ref[k])) for k in common]


# ── 2. Image-level split ──────────────────────────────────────────────────────
def split_pairs(
    pairs:       List[Tuple[str, str]],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = SEED,
) -> Tuple[list, list, list]:
    """
    Deterministic shuffle then split at image level.
    Test ratio = 1 - train_ratio - val_ratio  (≈ 0.15 by default).
    """
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n       = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = pairs[:n_train]
    val   = pairs[n_train : n_train + n_val]
    test  = pairs[n_train + n_val :]

    print(f"Split  →  train: {len(train)} | val: {len(val)} | test: {len(test)} images")
    return train, val, test


# ── 3. Load one image pair and extract patches (runs inside tf.py_function) ───
def _load_pair_to_patches(
    raw_path_eager: "tf.EagerTensor",
    ref_path_eager: "tf.EagerTensor",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads one (raw, reference) image pair, normalises to [0, 1],
    and extracts overlapping patches via patches.extract_patches.

    Called by tf.py_function — receives EagerTensors, calls .numpy() to decode.
    """
    raw_path = raw_path_eager.numpy().decode()
    ref_path = ref_path_eager.numpy().decode()

    raw = np.array(Image.open(raw_path).convert("RGB"), dtype=np.float32) / 255.0
    ref = np.array(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255.0

    return extract_patches(raw), extract_patches(ref)   # (N, 256, 256, 3) each


def _tf_load_pair(
    raw_path: tf.Tensor,
    ref_path: tf.Tensor,
) -> tf.data.Dataset:
    """
    flat_map target: load one image pair → Dataset of individual patch pairs.
    Each element is (patch_raw, patch_ref), both float32 (256, 256, 3).
    """
    p_raw, p_ref = tf.py_function(
        _load_pair_to_patches,
        [raw_path, ref_path],
        [tf.float32, tf.float32],
    )
    p_raw.set_shape([None, PATCH_SIZE, PATCH_SIZE, 3])
    p_ref.set_shape([None, PATCH_SIZE, PATCH_SIZE, 3])
    return tf.data.Dataset.from_tensor_slices((p_raw, p_ref))


# ── 4. Build a ready-to-use tf.data Dataset ───────────────────────────────────
def build_dataset(
    pairs:          List[Tuple[str, str]],
    batch_size:     int  = 8,
    shuffle:        bool = False,
    shuffle_buffer: int  = 500,
) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset of (patch_raw, patch_ref) batches.
    Shape per batch: (batch_size, 256, 256, 3), dtype float32.

    Args
    ----
    pairs          : list of (raw_path, ref_path) strings.
    batch_size     : number of patches per training step.
    shuffle        : shuffle image order AND patch order (use True for training).
    shuffle_buffer : size of the patch-level shuffle buffer.
    """
    raw_paths = [p[0] for p in pairs]
    ref_paths = [p[1] for p in pairs]

    path_ds = tf.data.Dataset.from_tensor_slices((raw_paths, ref_paths))

    if shuffle:
        path_ds = path_ds.shuffle(len(pairs), seed=SEED, reshuffle_each_iteration=True)

    patch_ds = path_ds.flat_map(_tf_load_pair)

    if shuffle:
        patch_ds = patch_ds.shuffle(shuffle_buffer, seed=SEED)

    return (
        patch_ds
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
