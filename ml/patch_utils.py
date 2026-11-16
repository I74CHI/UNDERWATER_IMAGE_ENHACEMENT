import math

import numpy as np

PATCH_SIZE = 256
OVERLAP = 32
STRIDE = PATCH_SIZE - OVERLAP

_axis_window = np.maximum(np.hanning(PATCH_SIZE), 1e-3)
_WINDOW = np.outer(_axis_window, _axis_window)


def _padding(length: int) -> int:
    steps = math.ceil(max(0, length - PATCH_SIZE) / STRIDE)
    return PATCH_SIZE + steps * STRIDE - length


def extract_patches(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or not all(image.shape):
        raise ValueError("Expected an image with shape (height, width, channels)")

    height, width, _ = image.shape
    pad_height, pad_width = _padding(height), _padding(width)
    mode = "reflect" if min(height, width) > 1 else "edge"
    padded = np.pad(
        image, ((0, pad_height), (0, pad_width), (0, 0)), mode=mode
    )
    return np.stack(
        [
            padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            for y in range(0, padded.shape[0] - PATCH_SIZE + 1, STRIDE)
            for x in range(0, padded.shape[1] - PATCH_SIZE + 1, STRIDE)
        ]
    )


def reconstruct_from_patches(
    patches: np.ndarray, orig_h: int, orig_w: int
) -> np.ndarray:
    if patches.ndim != 4 or patches.shape[1:3] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError("Expected patches with shape (count, 256, 256, channels)")

    pad_height, pad_width = _padding(orig_h), _padding(orig_w)
    padded_h, padded_w = orig_h + pad_height, orig_w + pad_width
    ys = range(0, padded_h - PATCH_SIZE + 1, STRIDE)
    xs = range(0, padded_w - PATCH_SIZE + 1, STRIDE)
    if len(patches) != len(ys) * len(xs):
        raise ValueError("Patch count does not match the requested image size")

    channels = patches.shape[3]
    canvas = np.zeros((padded_h, padded_w, channels), dtype=np.float64)
    weights = np.zeros((padded_h, padded_w), dtype=np.float64)
    window = _WINDOW.astype(np.float64)

    index = 0
    for y in ys:
        for x in xs:
            canvas[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += (
                patches[index] * window[:, :, None]
            )
            weights[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += window
            index += 1

    return (canvas / weights[:, :, None])[:orig_h, :orig_w].astype(np.float32)
