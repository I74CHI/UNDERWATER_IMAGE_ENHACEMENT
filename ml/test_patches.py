import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from patch_utils import extract_patches, reconstruct_from_patches


class PatchRoundTripTest(unittest.TestCase):
    def test_non_divisible_image_round_trip(self) -> None:
        image = np.random.default_rng(42).random((537, 813, 3), dtype=np.float32)
        patches = extract_patches(image)
        reconstructed = reconstruct_from_patches(patches, 537, 813)

        self.assertEqual(reconstructed.shape, image.shape)
        self.assertTrue(np.isfinite(reconstructed).all())
        error = float(np.max(np.abs(image - reconstructed)))
        self.assertLess(error, 1e-4, f"round-trip error was {error}")


if __name__ == "__main__":
    unittest.main()
