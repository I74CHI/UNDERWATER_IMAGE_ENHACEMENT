"""
patches.py — Shared patch extraction and reconstruction utilities.

Used by: dataset.py (training pipeline)
         inference.py  (deployment / backend)
         test_patch.py (tests)

Key design decisions
--------------------
1.  Reflect-pad the image before slicing so every pixel — including the
    right-most column and bottom row — is covered by at least one patch.
2.  Reconstruct via weighted averaging: each pixel accumulates the
    value from every patch that covers it, weighted by a 2-D cosine
    (Hanning) window that is highest at the patch centre and fades to
    zero at the edges.  Dividing by the total weight gives a smooth,
    seam-free result even at overlap boundaries.
"""

import math
import numpy as np

PATCH_SIZE = 256
OVERLAP    = 32
STRIDE     = PATCH_SIZE - OVERLAP   # 224 px


# ── 2-D blending window (flat centre with smooth cosine ramp at edges) ──────
def _blending_window_2d(size: int, overlap: int) -> np.ndarray:
    """
    Tukey-style window: 1.0 in the interior, cosine ramp across overlap margins.
    Clipped at 1e-3 so border weights are strictly positive, preventing division
    by zero or unweighted pixels at image edges.
    """
    w1d = np.ones(size, dtype=np.float32)
    t = np.linspace(0.0, np.pi, overlap, dtype=np.float32)
    ramp = np.clip(0.5 * (1.0 - np.cos(t)), 1e-3, 1.0).astype(np.float32)
    w1d[:overlap] = ramp
    w1d[-overlap:] = ramp[::-1]
    return np.outer(w1d, w1d).astype(np.float32)

_WINDOW = _blending_window_2d(PATCH_SIZE, OVERLAP)   # shape (256, 256)


# ── How many pixels to pad one image dimension ────────────────────────────────
def _pad_amount(dim: int) -> int:
    """
    Returns the number of pixels to pad so that patches of size PATCH_SIZE
    with stride STRIDE fit exactly (no leftover strip at the far edge).

    Works for any dim, including dim < PATCH_SIZE.
    """
    n = math.ceil(max(0, dim - PATCH_SIZE) / STRIDE)
    return PATCH_SIZE + n * STRIDE - dim


# ── Extract ───────────────────────────────────────────────────────────────────
def extract_patches(img: np.ndarray) -> np.ndarray:
    """
    Pad then slice `img` into overlapping 256×256 patches.

    Args
    ----
    img : float32 ndarray, shape (H, W, 3), values in [0, 1].

    Returns
    -------
    patches : float32 ndarray, shape (N, 256, 256, 3).
    """
    H, W, C = img.shape
    ph = _pad_amount(H)
    pw = _pad_amount(W)

    # reflect-pad — avoids introducing artificial dark/bright borders
    padded = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    pH, pW = padded.shape[:2]

    patches = []
    for y in range(0, pH - PATCH_SIZE + 1, STRIDE):
        for x in range(0, pW - PATCH_SIZE + 1, STRIDE):
            patches.append(padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE])

    return np.stack(patches, axis=0).astype(np.float32)


# ── Reconstruct ───────────────────────────────────────────────────────────────
def reconstruct_from_patches(
    patches: np.ndarray,
    orig_h:  int,
    orig_w:  int,
) -> np.ndarray:
    """
    Blend `patches` back into a full image of original size.

    Each pixel gets the weighted average of every patch that covers it.
    The cosine window weight ensures smooth transitions (no visible seams).
    Padding added during extraction is cropped off before returning.

    Args
    ----
    patches : float32, shape (N, 256, 256, C).
    orig_h  : original image height (before padding).
    orig_w  : original image width  (before padding).

    Returns
    -------
    img : float32 ndarray, shape (orig_h, orig_w, C).
    """
    C  = patches.shape[3]
    ph = _pad_amount(orig_h)
    pw = _pad_amount(orig_w)
    pH = orig_h + ph
    pW = orig_w + pw

    # float64 accumulators to avoid rounding drift
    canvas  = np.zeros((pH, pW, C), dtype=np.float64)
    weights = np.zeros((pH, pW),    dtype=np.float64)
    win     = _WINDOW.astype(np.float64)

    idx = 0
    for y in range(0, pH - PATCH_SIZE + 1, STRIDE):
        for x in range(0, pW - PATCH_SIZE + 1, STRIDE):
            canvas [y : y + PATCH_SIZE, x : x + PATCH_SIZE] += patches[idx] * win[:, :, None]
            weights[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += win
            idx += 1

    # Avoid division by zero (should not happen with correct padding, but safe)
    weights = np.where(weights == 0, 1.0, weights)
    result  = (canvas / weights[:, :, None]).astype(np.float32)

    return result[:orig_h, :orig_w]   # crop padding
