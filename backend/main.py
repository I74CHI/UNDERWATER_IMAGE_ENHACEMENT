"""
main.py — AquaEnhance FastAPI Backend Application.

Endpoints:
  POST /enhance   — Validate JPG/JPEG, enforce 5MB limit, process with U-Net,
                    save to storage/, insert row into Image table, return URLs + ID.
  POST /feedback — Validate 1–5 rating, verify image_id exists, store in Feedback.
  GET  /storage/* — Static files mount for original and enhanced images.
  GET  /health    — Quick health & DB check.
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── 1. Load environment variables from backend/.env ───────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import get_connection
from backend.services.enhancement_service import process_image

# ── 2. Configuration & Paths ──────────────────────────────────────────────────
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", PROJECT_ROOT / "backend" / "storage")).resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# ── 3. FastAPI App Setup ──────────────────────────────────────────────────────
app = FastAPI(
    title="AquaEnhance API",
    description="Underwater Image Enhancement System API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_PUBLIC = FRONTEND_DIR / "public"
if FRONTEND_PUBLIC.exists():
    app.mount("/public", StaticFiles(directory=str(FRONTEND_PUBLIC)), name="public")

@app.get("/")
def serve_index():
    """Serves the Victorian AquaEnhance web app."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AquaEnhance API is running."}


# ── 4. Models ─────────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    image_id: int
    rating: int = Field(..., ge=1, le=5, description="Quality rating from 1 to 5")


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: int
    image_id: int
    rating: int


# ── 5. Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Verify API and Database connectivity."""
    db_status = "unhealthy"
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {"status": "ok", "database": db_status}


@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    Accepts JPG/JPEG only (max 5 MB).
    Runs patch-based U-Net inference, records metadata in MSSQL,
    and returns URLs and image_id.
    """
    # ① MIME type validation
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG, JPEG, and PNG files are accepted.",
        )

    # ② Read and enforce 5 MB size limit
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File size exceeds maximum allowed limit of 5 MB.",
        )

    # ③ Magic bytes check for JPEG (0xFFD8) or PNG (\x89PNG\r\n\x1a\n)
    is_jpeg = content.startswith(b"\xff\xd8")
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a valid JPEG or PNG image.",
        )

    # ④ Generate unique filenames & persist to storage
    img_uid = uuid.uuid4().hex[:12]
    ext = ".png" if is_png else ".jpg"
    orig_filename = f"orig_{img_uid}{ext}"
    enh_filename = f"enh_{img_uid}{ext}"

    orig_path = STORAGE_DIR / orig_filename
    enh_path = STORAGE_DIR / enh_filename
    comparison_paths = [
        STORAGE_DIR / f"base_{method}_{img_uid}{ext}"
        for method in ("hist_eq", "clahe", "white_balance", "gamma", "retinex")
    ]
    generated_paths = [orig_path, enh_path, *comparison_paths]

    with open(orig_path, "wb") as f:
        f.write(content)

    # ⑤ Run decoupled enhancement
    try:
        proc_info = process_image(orig_path, enh_path)
    except Exception as exc:
        for path in generated_paths:
            if path.exists():
                path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image enhancement failed: {str(exc)}",
        )

    # ⑥ Log into Database
    try:
        conn = get_connection()
        cur = conn.cursor()
        driver = os.environ.get("DB_DRIVER", "mssql").lower()
        ph = "?" if driver == "mssql" else "%s"
        info_json = json.dumps(proc_info)

        if driver == "mssql":
            cur.execute(
                f"INSERT INTO Image (original_image, enhanced_image, processing_info) "
                f"OUTPUT INSERTED.image_id VALUES ({ph}, {ph}, {ph})",
                (orig_filename, enh_filename, info_json),
            )
            image_id = int(cur.fetchone()[0])
        else:
            cur.execute(
                f"INSERT INTO Image (original_image, enhanced_image, processing_info) "
                f"VALUES ({ph}, {ph}, {ph})",
                (orig_filename, enh_filename, info_json),
            )
            image_id = int(cur.lastrowid)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        for path in generated_paths:
            if path.exists():
                path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database record creation failed: {str(exc)}",
        )

    comparisons = [
        {
            "method": comparison["method"],
            "image_url": f"{API_BASE_URL}/storage/{comparison['filename']}",
            "metrics": comparison["metrics"],
        }
        for comparison in proc_info.pop("comparisons")
    ]

    return {
        "image_id": image_id,
        "original_url": f"{API_BASE_URL}/storage/{orig_filename}",
        "enhanced_url": f"{API_BASE_URL}/storage/{enh_filename}",
        "processing_info": proc_info,
        "comparisons": comparisons,
    }


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest):
    """
    Accepts image_id and rating (1–5) and stores in Feedback table.
    """
    conn = get_connection()
    cur = conn.cursor()
    driver = os.environ.get("DB_DRIVER", "mssql").lower()
    ph = "?" if driver == "mssql" else "%s"

    try:
        # Check that image_id exists
        cur.execute(f"SELECT image_id FROM Image WHERE image_id = {ph}", (payload.image_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image with id {payload.image_id} not found.",
            )

        if driver == "mssql":
            cur.execute(
                f"INSERT INTO Feedback (image_id, rating) OUTPUT INSERTED.feedback_id VALUES ({ph}, {ph})",
                (payload.image_id, payload.rating),
            )
            feedback_id = int(cur.fetchone()[0])
        else:
            cur.execute(
                f"INSERT INTO Feedback (image_id, rating) VALUES ({ph}, {ph})",
                (payload.image_id, payload.rating),
            )
            feedback_id = int(cur.lastrowid)

        conn.commit()
    finally:
        cur.close()
        conn.close()

    return FeedbackResponse(
        message="Feedback submitted successfully.",
        feedback_id=feedback_id,
        image_id=payload.image_id,
        rating=payload.rating,
    )
