"""
test_patch.py — Bug-reproduction test for patch extraction & reconstruction.

WHAT WE TEST
------------
Patch-based processing on images whose dimensions are NOT exact multiples
of the stride (224 px).  This is the most common real-world failure.

HOW IT WORKS
------------
Step 1  EXPECTED FAIL — demonstrate the bug
        A naive approach (no padding) skips the right/bottom pixel strips.
        Those pixels stay zero in the canvas → large pixel error.

Step 2  MUST PASS — verify our fix
        Our patches.py uses reflect-padding and cosine-window blending.
        The round-trip (extract → identity → reconstruct) must be near-perfect
        (max pixel error < 1e-4) for every image size, including awkward ones.

HOW TO RUN
----------
    cd UNDERWATER_IMAGE_ENHACEMENT
    python ml/test_patch.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from patches import PATCH_SIZE, STRIDE, extract_patches, reconstruct_from_patches

PASS = "✅ PASS"
FAIL = "🔴 EXPECTED FAIL (bug demonstrated)"
np.random.seed(42)


# ── Utility ───────────────────────────────────────────────────────────────────
def _synth(h: int, w: int) -> np.ndarray:
    """Random float32 image in [0, 1]."""
    return np.random.rand(h, w, 3).astype(np.float32)


# ── Step 1 : Show the bug ─────────────────────────────────────────────────────
def test_naive_fails() -> None:
    """
    BUG CASE
    --------
    Without padding, patches only cover rows 0..255 and cols 0..255.
    For a 347×523 image the last ~91 rows and ~267 cols are NEVER written
    to the canvas — they remain zero.

    Expected: max pixel error >> 0.1  (confirms bug is real)
    """
    H, W  = 347, 523
    img   = _synth(H, W)

    # Naive: extract only full-fit patches, no padding
    patches, positions = [], []
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            patches.append(img[y : y + PATCH_SIZE, x : x + PATCH_SIZE])
            positions.append((y, x))

    # Naive reconstruction — no blending, just overwrite
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for patch, (y, x) in zip(patches, positions):
        canvas[y : y + PATCH_SIZE, x : x + PATCH_SIZE] = patch

    max_err = float(np.max(np.abs(img - canvas)))

    if max_err > 0.1:
        print(
            f"{FAIL}  test_naive_fails\n"
            f"         → max pixel error = {max_err:.4f}  "
            f"(rightmost/bottom strips are unprocessed zeros)\n"
            f"         → This confirms the padding bug is real."
        )
    else:
        print(
            f"⚠️  UNEXPECTED: naive approach produced low error ({max_err:.4f}). "
            "Image may be coincidentally aligned to stride boundaries."
        )


# ── Step 2 : Verify our fix across multiple sizes ────────────────────────────
_CASES = [
    (256,  256,  "exact patch size (1 patch)"),
    (257,  257,  "one pixel over — forces padding + 4 patches"),
    (480,  640,  "VGA — stride-aligned"),
    (347,  523,  "non-divisible (the bug case from Step 1)"),
    (1080, 1920, "Full HD"),
    (100,  150,  "smaller than patch size"),
]

THRESHOLD = 1e-4   # max acceptable reconstruction error (float32 precision)


def test_padded_passes(h: int, w: int, label: str) -> bool:
    """
    FIX VERIFICATION
    ----------------
    extract_patches + reconstruct_from_patches is an identity transform
    when no model is applied (patches passed through unchanged).
    Max pixel error must be < THRESHOLD for the round-trip to pass.
    """
    img     = _synth(h, w)
    patches = extract_patches(img)
    recon   = reconstruct_from_patches(patches, h, w)

    max_err = float(np.max(np.abs(img - recon)))

    if max_err < THRESHOLD:
        print(
            f"{PASS}  [{h:4d}×{w:4d}]  {label}\n"
            f"         → max error = {max_err:.2e}"
        )
        return True
    else:
        print(
            f"❌ FAIL  [{h:4d}×{w:4d}]  {label}\n"
            f"         → max error = {max_err:.2e}  (threshold {THRESHOLD:.0e})"
        )
        return False


# ── Runner ────────────────────────────────────────────────────────────────────
def run_all() -> None:
    print(f"\nSettings  →  PATCH_SIZE={PATCH_SIZE}  OVERLAP=32  STRIDE={STRIDE}\n")

    print("═" * 65)
    print("Step 1 — Demonstrate the bug (naive approach, no padding)")
    print("═" * 65)
    test_naive_fails()

    print("\n" + "═" * 65)
    print("Step 2 — Verify the fix (padded + cosine blending)")
    print("═" * 65)
    results = [test_padded_passes(h, w, lbl) for h, w, lbl in _CASES]

    total  = len(results)
    passed = sum(results)

    print(f"\n{passed}/{total} reconstruction tests passed.")
    if passed == total:
        print(
            "\n✅ All patch tests passed — the pipeline handles any image size "
            "without seams or missing pixel strips."
        )
    else:
        print("\n❌ Some tests failed — review ml/patches.py.")


if __name__ == "__main__":
    run_all()
