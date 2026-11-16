"""
inference.py — Full-image U-Net inference pipeline.

Steps
-----
1. Load image (any size, any aspect ratio).
2. Normalise to float32 [0, 1].
3. Extract overlapping 256×256 patches (with reflect-padding).
4. Run the U-Net on each patch in mini-batches (saves VRAM).
5. Reconstruct the full image using cosine-window blending.
6. Crop padding and return the enhanced image as float32 [0, 1].

The model is loaded once and cached — subsequent calls on the same
model path reuse the cached instance (no cold-start delay).

Usage
-----
    from inference import enhance
    import numpy as np
    from PIL import Image

    enhanced = enhance("path/to/raw.jpg", "ml/model/best_model.keras")
    Image.fromarray((enhanced * 255).astype(np.uint8)).save("enhanced.jpg")
"""

import sys
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from patches import extract_patches, reconstruct_from_patches

# One cached model entry per model_path string
_model_cache: dict = {}


def _load_model(model_path: str) -> tf.keras.Model:
    """Load once, cache forever for the life of the process."""
    if model_path not in _model_cache:
        # compile=False: we only need forward-pass for inference,
        # so we skip reconstructing the custom combined_loss function.
        _model_cache[model_path] = tf.keras.models.load_model(
            model_path, compile=False
        )
        print(f"[inference] Model loaded from: {model_path}")
    return _model_cache[model_path]


def enhance(
    image_input: Union[str, Path, np.ndarray],
    model_path:  str = "ml/model/best_model.keras",
    batch_size:  int = 4,
) -> np.ndarray:
    """
    Enhance one underwater image end-to-end.

    Args
    ----
    image_input : file path  OR  float32 ndarray (H, W, 3) in [0, 1].
    model_path  : path to the trained .keras model file.
    batch_size  : patches processed per GPU call.
                  Reduce if you get OOM errors (try 2 or 1).

    Returns
    -------
    enhanced : float32 ndarray (H, W, 3) in [0, 1].
    """
    model = _load_model(str(model_path))

    # ── Load & normalise ──────────────────────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        raw = (
            np.array(Image.open(image_input).convert("RGB"), dtype=np.float32)
            / 255.0
        )
    else:
        raw = np.asarray(image_input, dtype=np.float32)
        if raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError(
                f"Expected shape (H, W, 3), got {raw.shape}"
            )

    orig_h, orig_w = raw.shape[:2]

    # ── Patch extraction ──────────────────────────────────────────────────────
    patches = extract_patches(raw)   # (N, 256, 256, 3)
    n       = len(patches)

    # ── Batched U-Net inference ───────────────────────────────────────────────
    enhanced_patches = np.empty_like(patches)
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = patches[start:end]
        enhanced_patches[start:end] = model.predict(batch, verbose=0)

    # ── Reconstruct full image ────────────────────────────────────────────────
    return reconstruct_from_patches(enhanced_patches, orig_h, orig_w)
