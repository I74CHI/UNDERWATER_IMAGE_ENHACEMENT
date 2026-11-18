# AquaEnhance — Project Memory, Specs & Strict Rules

## 1. Project Overview

- **Goal:** B.Tech final-year project — **AquaEnhance**, an automated underwater image enhancement web application (no user login/registration).
- **Workflow:** User uploads JPG/JPEG (max 5 MB) → Validate & Normalize → Slice into overlapping patches → U-Net inference → Overlap-blend patches into full image → Side-by-side Original vs. Enhanced display + optional manual Brightness & Sharpness sliders + Download + 1–5 human quality rating stored in DB.

## 2. Tech Stack & Module Specifications

- **Database (`database/`):** Microsoft SQL Server (`aquaenhance` DB via `pyodbc`, configured in `backend/.env` with `DB_DRIVER=mssql`, `DB_HOST=localhost`, `DB_PORT=1433`).
  - `Image` table: `image_id` (PK), `original_image` (file path), `enhanced_image` (file path), `upload_timestamp`, `processing_info` (JSON string).
  - `Feedback` table: `feedback_id` (PK), `image_id` (FK), `rating` (1–5), `timestamp`.
- **Deep Learning Model (`ml/`):** Python, TensorFlow/Keras standard **U-Net** (encoder-decoder with skip connections).
  - **Dataset Split:** 70% Train / 15% Validation / 15% Test (initial implementation without data augmentation).
  - **Patch Processing:** Slice large underwater images into smaller overlapping patches and reconstruct via overlap blending to prevent memory spikes and visible seams.
  - **Training & Loss:** Pixel/structural loss (L1/MAE, MSE, and/or SSIM) + overfitting controls (Dropout, L2 regularization, Early Stopping, Learning-Rate Scheduling). Must include a Google Colab/Kaggle-ready training script/notebook (no local GPU) exporting `.keras` or `.h5`.
  - **Evaluation & Baselines:** Compute PSNR, SSIM, MSE (paired) and UIQM, UCIQE (no-reference), plus baseline comparisons (Histogram Equalization, CLAHE, White Balance, Gamma Correction, Retinex).
- **Backend API (`backend/`):** Python, **FastAPI**.
  - `POST /enhance`: Accept JPG/JPEG/PNG, enforce 5 MB max size, normalize, patch-infer via U-Net, save files to `STORAGE_DIR`, log row in `Image`, return image URLs/data + `image_id`.
  - `POST /feedback`: Accept `image_id` and `1–5` integer rating, store in `Feedback` (no live retraining—stored for periodic offline analysis).
- **Frontend (`frontend/`):** **Next.js / React** (HTML, CSS, JS).
  - Upload form (JPG/JPEG/PNG, <=5 MB), side-by-side Original vs. Enhanced comparison, client-side Brightness & Sharpness adjustment sliders, Download button, 1–5 rating widget.

## 3. Current Project Status

- [x] **Phase 1: Database (`database/`)** — `schema.sql`, `db.py`, `backend/.env`, and `test_db.py` are complete and verified (all 4 DB tests pass on MSSQL `aquaenhance`). **DO NOT MODIFY.**
- [x] **Phase 2: Deep Learning Model (`ml/`)** — COMPLETE & VERIFIED.
  - Patch slicing & overlap blending with boundary padding (`patches.py`, `test_patch.py`, `test_patches.py` - all pass with 0.0 error).
  - U-Net architecture (`unet.py`).
  - Colab/Kaggle training script & notebook (`train.py`, `AquaEnhance_Train.ipynb`).
  - Evaluation metrics (PSNR, SSIM, MSE, UIQM, UCIQE) & baselines (HE, CLAHE, White Balance, Gamma, Retinex in `evaluate.py`).
  - Decoupled inference pipeline (`inference.py`).
- [x] **Phase 3: Backend API (`backend/`)** — COMPLETE & VERIFIED.
  - FastAPI app (`backend/main.py`) with `POST /enhance`, `POST /feedback`, `/storage` static mounts, and CORS.
  - Decoupled ML inference service (`backend/services/enhancement_service.py`).
  - Size enforcement (<= 5MB) and JPEG validation.
  - Integration with MSSQL `Image` and `Feedback` tables.
  - Automated verification test suite (`backend/test_api.py` - all 6 tests pass).
- [x] **Phase 4: Frontend Web App (`frontend/`)** — COMPLETE & VERIFIED.
  - Option A Dual Delivery implemented:
    - Zero-dependency standalone Victorian UI (`frontend/index.html`) served directly by FastAPI at `http://localhost:8000/`.
    - Next.js 14 App Router project (`frontend/package.json`, `app/page.js`, `app/layout.js`, `app/globals.css`).
  - Victorian aesthetics with PixelCode font (`frontend/public/fonts/`).
  - Client-side validation: JPG/JPEG only and <= 5 MB size enforcement.
  - Side-by-side Original vs. Enhanced viewer.
  - Client-side real-time Brightness & Sharpness (unsharp mask) canvas sliders.
  - Canvas-rendered JPEG download button.
  - 1–5 star interactive rating widget connected to `POST /feedback`.
  - Integration verified via automated test in `backend/test_api.py`.
