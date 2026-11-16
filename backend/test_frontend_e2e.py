"""
test_frontend_e2e.py — End-to-end verification of Frontend Web App & API.

Verifies:
  1. Frontend HTML at GET / includes all required UI components:
     - Header with Victorian styling and PixelCode typography
     - Dropzone with drag & drop and file input for JPG/JPEG/PNG <= 5MB
     - Original submerged frame viewer and Enhanced HTML5 Canvas
     - Brightness & Sharpness calibration sliders
     - Download Final Image button
     - 5-Star quality rating widget
  2. Offline PixelCode font assets load with HTTP 200.
  3. Uploading real user images through the frontend /enhance route:
     - Verifies processing statistics (patches, inference_ms, dimensions)
     - Verifies static image URLs can be fetched by the frontend
  4. Submitting ratings from the frontend 5-star widget to /feedback.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from database.db import get_connection

client = TestClient(app)

PASS = "✅ PASS"
FAIL = "❌ FAIL"

TEST_IMAGES = [
    r"C:/Users/I7ACHI/.gemini/antigravity/brain/3ce69719-4175-4cce-83ce-167cd27fe8f2/.user_uploaded/media_1794810596763_8e012a29.jpg",
    r"C:/Users/I7ACHI/.gemini/antigravity/brain/3ce69719-4175-4cce-83ce-167cd27fe8f2/.user_uploaded/media_1794810596779_c85c33ec.jpg",
    r"C:/Users/I7ACHI/.gemini/antigravity/brain/3ce69719-4175-4cce-83ce-167cd27fe8f2/.user_uploaded/media_1794810596797_dcc11142.jpg",
    r"C:/Users/I7ACHI/.gemini/antigravity/brain/3ce69719-4175-4cce-83ce-167cd27fe8f2/.user_uploaded/media_1794810596822_985b831a.jpg",
    r"C:/Users/I7ACHI/.gemini/antigravity/brain/3ce69719-4175-4cce-83ce-167cd27fe8f2/.user_uploaded/media_1794810596830_b213cbdf.jpg",
]

def run_e2e_tests():
    print("\n" + "=" * 65)
    print("E2E FRONTEND INTEGRATION & VERIFICATION TEST")
    print("=" * 65)

    # 1. Test HTML Structure & Elements
    print("\n1. Verifying Victorian Frontend HTML Delivery...")
    res = client.get("/")
    assert res.status_code == 200, "Frontend index.html failed to serve"
    html = res.text

    required_snippets = [
        "AquaEnhance",
        "PixelCode",
        "dropzone",
        "fileInput",
        "JPG / JPEG / PNG",
        "enhCanvas",
        "methodComparison",
        "UIQM",
        "UCIQE",
        "brightSlider",
        "sharpSlider",
        "downloadBtn",
        "starContainer",
        "data-rating=\"5\"",
    ]

    for snippet in required_snippets:
        assert snippet in html, f"Missing UI component in HTML: {snippet}"
    print(f"{PASS} All required Victorian UI components verified in DOM structure.")

    # 2. Test Fonts
    print("\n2. Verifying PixelCode Typography Assets...")
    font_res = client.get("/public/fonts/PixelCode.woff2")
    bold_font_res = client.get("/public/fonts/PixelCode-Bold.woff2")
    assert font_res.status_code == 200 and len(font_res.content) > 10000
    assert bold_font_res.status_code == 200 and len(bold_font_res.content) > 10000
    print(f"{PASS} Regular ({len(font_res.content)} bytes) & Bold ({len(bold_font_res.content)} bytes) PixelCode fonts served.")

    # 3. Test Real User Image Ingestion Through Frontend Flow
    print("\n3. Testing Frontend Image Upload & Enhancement Flow on Real Images...")
    created_ids = []

    for idx, img_path in enumerate(TEST_IMAGES, 1):
        p = Path(img_path)
        with open(p, "rb") as f:
            file_bytes = f.read()

        upload_res = client.post(
            "/enhance",
            files={"file": (p.name, file_bytes, "image/jpeg")},
        )
        assert upload_res.status_code == 200, f"Upload failed for image {idx}: {upload_res.text}"
        data = upload_res.json()
        img_id = data["image_id"]
        created_ids.append(img_id)
        assert len(data["comparisons"]) == 6
        assert [item["method"] for item in data["comparisons"][1:]] == [
            "Histogram Equalization",
            "CLAHE",
            "White Balance",
            "Gamma Correction",
            "Retinex-based Enhancement",
        ]

        # Verify frontend can fetch original and enhanced images
        orig_fetch = client.get(data["original_url"])
        enh_fetch = client.get(data["enhanced_url"])
        assert orig_fetch.status_code == 200, f"Frontend failed to load original image URL"
        assert enh_fetch.status_code == 200, f"Frontend failed to load enhanced image URL"

        patches = data["processing_info"]["patches"]
        t_ms = data["processing_info"]["inference_ms"]
        dims = data["processing_info"]["dimensions"]
        print(f"{PASS} Image {idx} [{dims}] processed ({patches} patches, {t_ms}ms) -> Render URLs loaded successfully.")

    # 4. Test 5-Star Rating Feedback From Frontend
    print("\n4. Testing Frontend 5-Star Rating Submission...")
    for idx, img_id in enumerate(created_ids, 1):
        rating_val = min(5, idx + 1)
        fb_res = client.post(
            "/feedback",
            json={"image_id": img_id, "rating": rating_val},
        )
        assert fb_res.status_code == 200, f"Feedback failed: {fb_res.text}"
        fb_data = fb_res.json()
        assert fb_data["rating"] == rating_val
    print(f"{PASS} All ratings (2★, 3★, 4★, 5★, 5★) successfully submitted & logged to database.")

    # 5. Database Verification & Cleanup
    print("\n5. Verifying Database Records & Cleaning Up Test Entries...")
    conn = get_connection()
    cur = conn.cursor()
    driver = sys.modules['os'].environ.get("DB_DRIVER", "mssql").lower()
    ph = "?" if driver == "mssql" else "%s"

    placeholders = ", ".join([ph] * len(created_ids))
    cur.execute(f"SELECT COUNT(*) FROM Feedback WHERE image_id IN ({placeholders})", tuple(created_ids))
    fb_count = cur.fetchone()[0]
    assert fb_count == len(created_ids), f"Expected {len(created_ids)} feedback rows, found {fb_count}"
    print(f"{PASS} Confirmed {fb_count} feedback rows stored in Microsoft SQL Server.")

    cur.execute(f"DELETE FROM Image WHERE image_id IN ({placeholders})", tuple(created_ids))
    conn.commit()
    cur.close()
    conn.close()
    print(f"{PASS} Database cleaned up.")

    print("\n" + "=" * 65)
    print("✅ FRONTEND E2E TEST COMPLETE — ALL CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_e2e_tests()
