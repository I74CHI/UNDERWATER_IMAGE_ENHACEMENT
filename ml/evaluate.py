"""
evaluate.py — Metrics and baseline comparisons for AquaEnhance.

Paired metrics (need ground-truth reference image):
    psnr  — Peak Signal-to-Noise Ratio     (dB, higher = better)
    ssim  — Structural Similarity Index    (0-1, higher = better)
    mse   — Mean Squared Error             (lower = better)

No-reference underwater metrics (no ground truth needed):
    uiqm  — Underwater Image Quality Measure   (Panetta et al., 2016)
    uciqe — Underwater Color Image Quality Eval (Yang et al., 2015)

Baselines (classical methods for comparison against U-Net):
    hist_eq       — Histogram Equalisation (per RGB channel)
    clahe         — CLAHE on LAB L-channel (better colour preservation)
    white_balance — Gray World assumption
    gamma         — Power-law brightness boost
    retinex       — Single-Scale Retinex (colour constancy)

Usage:
    from evaluate import compute_all_metrics, baseline_enhance
    metrics = compute_all_metrics(enhanced_img, reference=ref_img)
    baselines = baseline_enhance(raw_img)   # dict of {name: enhanced_img}
"""

import math
import numpy as np
import cv2
from skimage.metrics import (
    structural_similarity as _sk_ssim,
    peak_signal_noise_ratio as _sk_psnr,
)


# ══════════════════════════════════════════════════════════════════════════════
# Paired metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(img: np.ndarray, ref: np.ndarray) -> float:
    """PSNR in dB. Both float32 [0,1], same shape."""
    if np.array_equal(img, ref):
        return 100.0
    return float(_sk_psnr(ref, img, data_range=1.0))


def compute_ssim(img: np.ndarray, ref: np.ndarray) -> float:
    """SSIM scalar. Both float32 [0,1], same shape."""
    return float(_sk_ssim(ref, img, data_range=1.0, channel_axis=2))


def compute_mse(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
# No-reference underwater metrics — internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _uicm(img: np.ndarray) -> float:
    """
    Underwater Image Colorfulness Measure.
    Uses alpha-trimmed mean & std of opponent colour channels RG and YB.
    """
    img_f = img.astype(np.float64)
    RG = img_f[:, :, 0] - img_f[:, :, 1]
    YB = (img_f[:, :, 0] + img_f[:, :, 1]) / 2.0 - img_f[:, :, 2]

    def _atm(arr: np.ndarray, alpha: float = 0.1):
        flat   = np.sort(arr.flatten())
        n      = len(flat)
        t      = max(1, int(alpha * n / 2))
        trimmed = flat[t : n - t] if n > 2 * t else flat
        return trimmed.mean(), trimmed.std()

    mu_rg, s_rg = _atm(RG)
    mu_yb, s_yb = _atm(YB)
    return (
        -0.0268 * math.sqrt(mu_rg ** 2 + mu_yb ** 2)
        + 0.1586 * math.sqrt(s_rg ** 2 + s_yb ** 2)
    )


def _eme(channel: np.ndarray, r: int = 8) -> float:
    """
    Enhancement Measure Estimation on a single-channel (HxW) image.
    Splits into r×r blocks and sums log(max/min) over all blocks.
    """
    H, W = channel.shape
    bh, bw = H // r, W // r
    if bh == 0 or bw == 0:
        return 0.0
    score = 0.0
    for i in range(r):
        for j in range(r):
            block = channel[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            mn, mx = block.min(), block.max()
            if mn > 1e-5:
                score += math.log(mx / mn)
    return 2.0 * score / (r * r)


def _uism(img: np.ndarray) -> float:
    """
    Underwater Image Sharpness Measure.
    Applies Sobel edge detection per channel, then EME, weighted by luminance.
    """
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    scores = []
    for c in range(3):
        ch    = u8[:, :, c].astype(np.float64)
        sobel = (
            np.abs(cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3))
            + np.abs(cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3))
        )
        scores.append(_eme(sobel))
    # Luminance-weighted combination (ITU-R BT.601)
    return 0.299 * scores[0] + 0.587 * scores[1] + 0.114 * scores[2]


def _uiconm(img: np.ndarray, r: int = 8) -> float:
    """
    Underwater Image Contrast Measure using LogAMEE.
    Operates on the grayscale image split into r×r blocks.
    """
    gray = img.mean(axis=2)
    H, W = gray.shape
    bh, bw = H // r, W // r
    if bh == 0 or bw == 0:
        return 0.0
    score = 0.0
    for i in range(r):
        for j in range(r):
            block = gray[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            mn, mx = block.min(), block.max()
            denom  = mx + mn
            if denom > 1e-5:
                amee = (mx - mn) / denom
                if amee > 1e-5:
                    score += amee * math.log(amee)
    return -score / (r * r)


# ══════════════════════════════════════════════════════════════════════════════
# No-reference underwater metrics — public API
# ══════════════════════════════════════════════════════════════════════════════

def uiqm(img: np.ndarray) -> float:
    """
    Underwater Image Quality Measure (Panetta et al., 2016).
    img : float32 RGB in [0, 1], shape (H, W, 3).
    Returns a scalar — higher is better.
    """
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(img)


def uciqe(img: np.ndarray) -> float:
    """
    Underwater Color Image Quality Evaluation (Yang et al., 2015).
    Uses CIELAB chroma std, L-channel Michelson contrast, and HSV saturation mean.
    img : float32 RGB in [0, 1], shape (H, W, 3).
    Returns a scalar — higher is better.
    """
    u8  = (img * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float64)
    L, a, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    sigma_c = np.sqrt(a ** 2 + b_ch ** 2).std()          # chroma std

    l_max, l_min = L.max(), L.min()
    con_l = (l_max - l_min) / (l_max + l_min + 1e-5)     # Michelson contrast

    hsv  = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float64)
    mu_s = hsv[:, :, 1].mean() / 255.0                   # mean saturation [0,1]

    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


# ══════════════════════════════════════════════════════════════════════════════
# Combined report
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    enhanced:  np.ndarray,
    reference: np.ndarray = None,
) -> dict:
    """
    Compute all available metrics for one enhanced image.

    Args
    ----
    enhanced  : float32 RGB in [0, 1].
    reference : float32 RGB in [0, 1], same shape, or None for no-reference only.

    Returns
    -------
    dict with keys: uiqm, uciqe (always) + psnr, ssim, mse (if reference given).
    """
    result = {
        "uiqm":  uiqm(enhanced),
        "uciqe": uciqe(enhanced),
    }
    if reference is not None:
        result["psnr"] = compute_psnr(enhanced, reference)
        result["ssim"] = compute_ssim(enhanced, reference)
        result["mse"]  = compute_mse(enhanced,  reference)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Baseline enhancers
# ══════════════════════════════════════════════════════════════════════════════

def _u8(img: np.ndarray) -> np.ndarray:
    return (img * 255).clip(0, 255).astype(np.uint8)

def _f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def hist_eq(img: np.ndarray) -> np.ndarray:
    """Histogram Equalisation independently on each RGB channel."""
    u8  = _u8(img)
    out = np.stack([cv2.equalizeHist(u8[:, :, c]) for c in range(3)], axis=2)
    return _f32(out)


def clahe_enhance(
    img:        np.ndarray,
    clip_limit: float = 2.0,
    tile:       int   = 8,
) -> np.ndarray:
    """
    CLAHE on the LAB L-channel only.
    Boosts local contrast without over-saturating colours.
    """
    u8  = _u8(img)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return _f32(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def white_balance(img: np.ndarray) -> np.ndarray:
    """
    Gray World white balance.
    Assumes the average colour of the scene should be neutral grey.
    Corrects the blue-green underwater colour cast.
    """
    img_f = img.astype(np.float64)
    mu    = img_f.mean(axis=(0, 1))       # (3,) per-channel mean
    grand = mu.mean()                      # overall mean
    out   = (img_f * (grand / (mu + 1e-7))).clip(0, 1)
    return out.astype(np.float32)


def gamma_correct(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Power-law gamma correction: output = input^(1/gamma).
    gamma > 1 brightens dark underwater images.
    """
    return np.power(img.clip(0, 1), 1.0 / gamma).astype(np.float32)


def retinex(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """
    Single-Scale Retinex (SSR) for colour constancy and contrast enhancement.
    log(I) - log(GaussianBlur(I))  per channel, then normalise to [0,1].
    """
    img_f = img.astype(np.float64) + 1e-6
    out   = np.zeros_like(img_f)
    for c in range(3):
        blurred          = cv2.GaussianBlur(img_f[:, :, c], (0, 0), sigma)
        out[:, :, c]     = np.log(img_f[:, :, c]) - np.log(blurred + 1e-6)
    out -= out.min()
    if out.max() > 1e-5:
        out /= out.max()
    return out.astype(np.float32)


def land_like_restore(img: np.ndarray) -> np.ndarray:
    """
    Physics-informed underwater optical recovery pipeline (Land-Equivalent Mode):
    1. Red/Blue channel attenuation compensation (inverting Beer-Lambert absorption).
    2. Color constancy / Gray-World balance (removes cyan/green water haze).
    3. Dynamic range percentile contrast stretching.
    4. CLAHE on CIELAB Luminance (recovers fine underwater textures).
    5. Natural vibrance boost in HSV.
    """
    res = img.copy().astype(np.float32)

    # 1. Red channel attenuation compensation (water absorbs red 10x faster than blue)
    r, g, b = res[:, :, 0], res[:, :, 1], res[:, :, 2]
    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))

    red_gain = max(0.0, (mean_g - mean_r)) * 1.3
    r_comp = r + red_gain * (1.0 - r) * g
    res[:, :, 0] = np.clip(r_comp, 0.0, 1.0)

    if mean_g > mean_b:
        blue_gain = max(0.0, (mean_g - mean_b)) * 0.7
        b_comp = b + blue_gain * (1.0 - b) * g
        res[:, :, 2] = np.clip(b_comp, 0.0, 1.0)

    # 2. White Balance / Color Constancy
    means = np.mean(res, axis=(0, 1))
    target = np.mean(means)
    scale = target / (means + 1e-6)
    res = np.clip(res * scale, 0.0, 1.0)

    # 3. Dynamic Range Histogram Stretching
    for c in range(3):
        low, high = np.percentile(res[:, :, c], (0.8, 99.2))
        if high > low:
            res[:, :, c] = np.clip((res[:, :, c] - low) / (high - low), 0.0, 1.0)

    # 4. Local Contrast in CIELAB
    u8 = (res * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge([l_enh, a, b_ch])
    rgb_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # 5. Natural Vibrance / Saturation boost in HSV
    u8_enh = (rgb_enh * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(u8_enh, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.30, 0.0, 255.0)
    final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return np.clip(final, 0.0, 1.0)


# ── Combined baseline dict ────────────────────────────────────────────────────
BASELINES = {
    "hist_eq":       hist_eq,
    "clahe":         clahe_enhance,
    "white_balance": white_balance,
    "gamma":         gamma_correct,
    "retinex":       retinex,
    "land_like":     land_like_restore,
}


def baseline_enhance(img: np.ndarray) -> dict:
    """Apply all 5 baselines. Returns {name: enhanced_img_float32}."""
    return {name: fn(img) for name, fn in BASELINES.items()}
