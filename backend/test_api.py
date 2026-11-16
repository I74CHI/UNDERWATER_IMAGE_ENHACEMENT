"""
test_api.py — Phase 3 Backend API Test Suite.

Tests:
  1. Oversized file (> 5 MB)  → rejected with HTTP 413.
  2. Non-JPEG / invalid file  → rejected with HTTP 400.
  3. Invalid rating (0 or 6)  → rejected with HTTP 422.
  4. Non-existent image_id    → rejected with HTTP 404.
  5. Valid JPEG enhance flow  → returns 200, saves files, creates Image row.
  6. Valid feedback flow      → returns 200, creates Feedback row.

Run:
    python backend/test_api.py
"""

import io
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from database.db import get_connection

client = TestClient(app)

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def create_dummy_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_dummy_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image in memory (RGBA supported)."""
    arr = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_tests():
    print("\n" + "=" * 60)
    print("Running Phase 3 Backend API Tests")
    print("=" * 60)

    # ── Test 1: Oversized file (> 5 MB) ──────────────────────────────────────
    print("\n1. Testing Oversized File (> 5 MB) Rejection...")
    oversized_data = b"\xff\xd8" + b"0" * (5 * 1024 * 1024 + 100)
    res = client.post(
        "/enhance",
        files={"file": ("large.jpg", oversized_data, "image/jpeg")},
    )
    if res.status_code == 413:
        print(f"{PASS} Correctly rejected oversized file with HTTP 413.")
    else:
        print(f"{FAIL} Expected 413, got {res.status_code}: {res.text}")
        assert res.status_code == 413

    # ── Test 2: Invalid format / Non-JPEG rejection ────────────────────────────
    print("\n2. Testing Non-JPEG File Rejection...")
    res = client.post(
        "/enhance",
        files={"file": ("doc.txt", b"Hello text file content", "text/plain")},
    )
    if res.status_code == 400:
        print(f"{PASS} Correctly rejected non-JPEG MIME with HTTP 400.")
    else:
        print(f"{FAIL} Expected 400, got {res.status_code}: {res.text}")
        assert res.status_code == 400

    # Test PNG mime or fake extension
    res2 = client.post(
        "/enhance",
        files={"file": ("fake.jpg", b"not-a-jpeg-header-bytes", "image/jpeg")},
    )
    if res2.status_code == 400:
        print(f"{PASS} Correctly rejected fake JPEG magic bytes with HTTP 400.")
    else:
        print(f"{FAIL} Expected 400, got {res2.status_code}: {res2.text}")
        assert res2.status_code == 400

    # ── Test 3: Invalid feedback ratings ──────────────────────────────────────
    print("\n3. Testing Invalid Feedback Ratings (Rating Out-of-Bounds)...")
    res_rating_0 = client.post("/feedback", json={"image_id": 1, "rating": 0})
    res_rating_6 = client.post("/feedback", json={"image_id": 1, "rating": 6})
    if res_rating_0.status_code == 422 and res_rating_6.status_code == 422:
        print(f"{PASS} Correctly rejected ratings 0 and 6 with HTTP 422.")
    else:
        print(f"{FAIL} Expected 422, got {res_rating_0.status_code} and {res_rating_6.status_code}")
        assert res_rating_0.status_code == 422 and res_rating_6.status_code == 422

    # ── Test 4: Feedback for non-existent image_id ────────────────────────────
    print("\n4. Testing Feedback For Non-Existent image_id...")
    res_orphan = client.post("/feedback", json={"image_id": 99999999, "rating": 4})
    if res_orphan.status_code == 404:
        print(f"{PASS} Correctly returned HTTP 404 for non-existent image_id.")
    else:
        print(f"{FAIL} Expected 404, got {res_orphan.status_code}: {res_orphan.text}")
        assert res_orphan.status_code == 404

    # ── Test 5: Valid JPEG enhancement flow ───────────────────────────────────
    print("\n5. Testing Valid /enhance Flow...")
    jpeg_bytes = create_dummy_jpeg(300, 200)
    res_enhance = client.post(
        "/enhance",
        files={"file": ("underwater_test.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert res_enhance.status_code == 200, f"Enhance failed: {res_enhance.text}"
    data = res_enhance.json()
    image_id = data["image_id"]
    print(f"{PASS} Enhance successful: image_id={image_id}, processing={data['processing_info']}")
    assert "original_url" in data and "enhanced_url" in data
    assert data["processing_info"]["patches"] >= 1
    expected_methods = [
        "Histogram Equalization",
        "CLAHE",
        "White Balance",
        "Gamma Correction",
        "Retinex-based Enhancement",
    ]
    assert data["comparisons"][0]["method"].startswith("AquaEnhance (")
    assert [item["method"] for item in data["comparisons"][1:]] == expected_methods
    for comparison in data["comparisons"]:
        assert set(comparison["metrics"]) == {"uiqm", "uciqe"}
        assert client.get(comparison["image_url"]).status_code == 200
    print(f"{PASS} Enhanced output and five classical baselines include UIQM/UCIQE scores and loadable images.")

    # Verify static file download
    orig_res = client.get(data["original_url"])
    enh_res = client.get(data["enhanced_url"])
    assert orig_res.status_code == 200, "Original image static endpoint failed"
    assert enh_res.status_code == 200, "Enhanced image static endpoint failed"
    print(f"{PASS} Static files served properly via /storage mount.")

    # ── Test 6: Valid feedback submission ─────────────────────────────────────
    print("\n6. Testing Valid /feedback Submission...")
    res_fb = client.post("/feedback", json={"image_id": image_id, "rating": 5})
    assert res_fb.status_code == 200, f"Feedback failed: {res_fb.text}"
    fb_data = res_fb.json()
    print(f"{PASS} Feedback registered: feedback_id={fb_data['feedback_id']}, rating={fb_data['rating']}")

    # ── Test 7: Frontend Web App Serving ──────────────────────────────────────
    print("\n7. Testing Frontend Web App & Static Font Serving...")
    res_front = client.get("/")
    assert res_front.status_code == 200, f"Frontend failed: {res_front.status_code}"
    assert "AquaEnhance" in res_front.text
    print(f"{PASS} Frontend HTML successfully served at GET /.")

    res_font = client.get("/public/fonts/PixelCode.woff2")
    assert res_font.status_code == 200, f"Font failed: {res_font.status_code}"
    print(f"{PASS} PixelCode webfont successfully served at /public/fonts/PixelCode.woff2.")

    # ── Test 8: Valid PNG enhancement flow ───────────────────────────────────
    print("\n8. Testing Valid PNG /enhance Flow...")
    png_bytes = create_dummy_png(250, 180)
    res_png = client.post(
        "/enhance",
        files={"file": ("underwater_specimen.png", png_bytes, "image/png")},
    )
    assert res_png.status_code == 200, f"PNG Enhance failed: {res_png.text}"
    png_data = res_png.json()
    png_image_id = png_data["image_id"]
    print(f"{PASS} PNG Enhance successful: image_id={png_image_id}, url={png_data['enhanced_url']}")
    assert png_data["enhanced_url"].endswith(".png")

    # ── Cleanup created test rows from DB ─────────────────────────────────────
    conn = get_connection()
    cur = conn.cursor()
    driver = os.environ.get("DB_DRIVER", "mssql").lower()
    ph = "?" if driver == "mssql" else "%s"
    cur.execute(f"DELETE FROM Image WHERE image_id IN ({ph}, {ph})", (image_id, png_image_id))
    conn.commit()
    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print("✅ All 8 Backend API tests (including PNG) passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
