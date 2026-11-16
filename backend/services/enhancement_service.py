"""
enhancement_service.py — Decoupled ML inference service.

Handles:
  1. Patch extraction and reconstruction via ml/patches.py
  2. U-Net inference via ml/inference.py (if trained model exists)
  3. Seamless CLAHE baseline fallback if model weights are not yet deployed
  4. Classical baseline images and no-reference quality metrics
  5. Timing and patch statistics logging
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image

# Add project root to sys.path so ml/ and database/ modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.patches import extract_patches, reconstruct_from_patches
from ml.evaluate import BASELINES, compute_all_metrics, land_like_restore

MODEL_PATH = PROJECT_ROOT / "ml" / "model" / "best_model.keras"


def process_image(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Load raw image, generate the enhanced and classical baseline results,
    save the images, and return processing metadata with no-reference metrics.
    """
    start_time = time.perf_counter()

    raw_pil = Image.open(input_path).convert("RGB")
    raw_np = np.asarray(raw_pil, dtype=np.float32) / 255.0
    orig_h, orig_w = raw_np.shape[:2]

    patches = extract_patches(raw_np)
    patch_count = len(patches)

    method = "unet"
    if MODEL_PATH.exists():
        try:
            from ml.inference import enhance
            enhanced_np = enhance(raw_np, str(MODEL_PATH))
        except Exception:
            # If TF or model load fails, fall back to physics-informed land-like restoration
            enhanced_np = land_like_restore(raw_np)
            method = "land_like_optical"
    else:
        # Development fallback: physics-informed optical restoration (compensates red attenuation,
        # removes cyan/green veil, enhances contrast & vibrance as if taken on land)
        enhanced_np = land_like_restore(raw_np)
        method = "land_like_optical"

    # Ensure output is clipped to [0, 1] and saved with correct format
    enhanced_uint8 = (np.clip(enhanced_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_format = "PNG" if output_path.suffix.lower() == ".png" else "JPEG"
    save_kwargs = {"quality": 95} if out_format == "JPEG" else {}
    Image.fromarray(enhanced_uint8).save(output_path, format=out_format, **save_kwargs)

    comparison_methods = (
        ("hist_eq", "Histogram Equalization"),
        ("clahe", "CLAHE"),
        ("white_balance", "White Balance"),
        ("gamma", "Gamma Correction"),
        ("retinex", "Retinex-based Enhancement"),
    )
    image_id = output_path.stem[4:] if output_path.stem.startswith("enh_") else output_path.stem
    comparisons = []
    for baseline_key, label in comparison_methods:
        baseline_np = BASELINES[baseline_key](raw_np)
        baseline_uint8 = (np.clip(baseline_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        baseline_kwargs = {"quality": 95} if out_format == "JPEG" else {}
        filename = f"base_{baseline_key}_{image_id}{output_path.suffix}"
        Image.fromarray(baseline_uint8).save(
            output_path.with_name(filename),
            format=out_format,
            **baseline_kwargs,
        )
        comparisons.append(
            {
                "method": label,
                "filename": filename,
                "metrics": {
                    name: float(value)
                    for name, value in compute_all_metrics(baseline_np).items()
                },
            }
        )

    enhancement_label = "AquaEnhance (U-Net)" if method == "unet" else "AquaEnhance (Optical)"
    comparisons.insert(
        0,
        {
            "method": enhancement_label,
            "filename": output_path.name,
            "metrics": {
                name: float(value)
                for name, value in compute_all_metrics(enhanced_np).items()
            },
        },
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "patches": patch_count,
        "inference_ms": elapsed_ms,
        "method": method,
        "dimensions": f"{orig_w}x{orig_h}",
        "comparisons": comparisons,
    }
