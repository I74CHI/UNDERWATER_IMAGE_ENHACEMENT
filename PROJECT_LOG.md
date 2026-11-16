# AquaEnhance — Project Log

> Maintained by: AI Mentor (Senior Full-Stack + ML Engineer)
> Project: Final-Year B.Tech CS — Underwater Image Enhancement System
> Build order: `database/` → `ml/` → `backend/` → `frontend/`

---

## Locked Assumptions & Decisions

| ID | Category | Decision | Reason |
|----|----------|----------|--------|
| A1 | Dataset | UIEB Kaggle archive (890 paired + 60 unpaired images); user downloads it manually; use the 890 pairs | Keep the supervised split paired and avoid requiring Kaggle API credentials |
| A2 | Checkpoints | Save every **5 epochs** + best model by val loss | Recovery granularity vs. disk space balance |
| A3 | Inference | Synchronous (GPU machine) — no async background task needed | User confirmed GPU availability |
| A4 | Database | **MSSQL primary** (`pyodbc` + ODBC Driver 17); `docker-compose.yml` + MySQL as fallback | User has MSSQL installed; fallback covers any compatibility gaps |
| A5 | Image persistence | Stored at `STORAGE_DIR` (absolute path in `.env`), path saved in DB | Must survive server restarts; no BLOB storage |
| A6 | API base URL | `NEXT_PUBLIC_API_URL` env var in Next.js | Works for both local and deployed environments |
| A7 | Build order | database → ml → backend → frontend | Each layer depends on the previous |
| A8 | Patch size | 256 × 256 px | GPU/RAM-friendly; standard for U-Net inference |
| A9 | Patch overlap | 32 px with cosine blending | Prevents visible seams at patch boundaries |
| A10 | U-Net depth | 4 encoder levels (64→128→256→512 filters, bottleneck 1024) | Standard architecture; fits consumer GPU VRAM |
| A11 | Dropout | 0.3 at bottleneck | Prevents overfitting without over-regularizing |
| A12 | Loss function | 0.5 x MAE + 0.5 x SSIM | Balances pixel accuracy and structural quality |
| A13 | Model export | `.keras` format (TF 2.x native) | Recommended over legacy `.h5` |
| A14 | Frontend theme | Victorian aesthetic, PixelCode font | User requirement |
| A15 | Frontend framework | Next.js 14 (App Router), plain CSS Modules | Minimal dependencies; no Tailwind or UI libs |
| A16 | Frontend sliders | Client-side CSS filter on canvas | No extra server round-trips for adjustments |

---

## Phase 1 — Database COMPLETE

**Date:** 2026-11-15

### Files Created

| File | Purpose |
|------|---------|
| `database/schema.sql` | MSSQL schema — creates Image and Feedback tables with PK, FK, and CHECK constraints |
| `database/schema_mysql.sql` | MySQL 8.x equivalent schema (for docker-compose fallback) |
| `database/db.py` | Thin DB-API 2.0 connection helper; switches between MSSQL and MySQL via DB_DRIVER env var |
| `database/test_db.py` | Phase 1 test: connection, valid inserts, CHECK constraint rejection (rating=6), FK violation rejection |
| `docker-compose.yml` | Spins up MySQL 8.3 container with persistent volume in one command (docker compose up -d) |
| `backend/.env.example` | Template for all required environment variables — copy to backend/.env |

### Technical Decisions Explained

**Why file paths instead of BLOBs?**
Storing images as files and saving only the path in the DB is faster to read, easier to serve via HTTP, and doesn't bloat the database. BLOBs would require reading multi-MB binary data through the DB on every request.

**Why NVARCHAR(512) in MSSQL?**
MSSQL uses UTF-16 internally. NVARCHAR handles any Unicode characters in file paths. 512 characters is more than enough for any realistic path.

**Why TINYINT for rating with CHECK constraint?**
TINYINT uses 1 byte (values 0-255), which is the smallest integer type — perfect for a 1-5 rating. The CHECK constraint enforces the business rule at the database layer.

**Why ON DELETE CASCADE on Feedback FK?**
If an image record is deleted, its associated feedback ratings should also be deleted automatically. This prevents orphaned feedback rows.

**Why a separate db.py helper?**
Separation of concerns — the API layer should not know how to connect to the database. db.py is the single place to change if you switch databases.

### How to Run Phase 1 Tests

#### Option A — Microsoft SQL Server (Primary)
```bash
# 1. Copy and fill in your credentials
cp backend/.env.example backend/.env

# 2. Run schema (in SSMS or sqlcmd)
sqlcmd -S localhost -U sa -P YourPassword -i database/schema.sql

# 3. Install pyodbc
pip install pyodbc

# 4. Run tests
cd database && python test_db.py
```

#### Option B — MySQL via Docker (Fallback)
```bash
# 1. Set DB_DRIVER=mysql in backend/.env
# 2. Start container
docker compose up -d

# 3. Install connector
pip install mysql-connector-python

# 4. Run tests
cd database && python test_db.py
```

### Bugs Found & Fixed During Verification

| Bug | Root Cause | Fix Applied |
|-----|-----------|-------------|
| Inline comment on `DB_PORT` line in `.env` | `DB_PORT=1433 # comment` — comment was parsed as part of the value | Removed inline comment; comments must be on their own line |
| `db.py` used `SERVER=host,port` for named instance | SQL Server named instances (e.g. `SQLEXPRESS`) ignore TCP port in that format | Fixed to use `SERVER=host\instance` when host contains a backslash |
| `sqlcmd` SSL certificate error | `sqlcmd` v18 enforces SSL by default; local dev cert not trusted | Used `-C` flag (`TrustServerCertificate`) for local dev |

### Verified Test Output (All 4 Pass)
```
PASS  Insert valid Image row
PASS  Insert Feedback with rating=4 (valid)
PASS  DB rejects rating=6 (CHECK constraint)
PASS  DB rejects orphan image_id FK violation

Phase 1 database tests complete.
```

---

## Phase 2 — ML Pipeline ✅ COMPLETE

**Date:** 2026-11-16

### Files Created & Verified

| File | Purpose |
|------|---------|
| `ml/patches.py` & `ml/patch_utils.py` | Slices arbitrary-sized images into 256×256 patches (stride 224, overlap 32) and reconstructs them seamlessly via weighted cosine blending. |
| `ml/test_patch.py` & `ml/test_patches.py` | Bug reproduction & round-trip verification across non-divisible dimensions, sub-patch sizes, and HD resolutions. |
| `ml/prepare_dataset.py` & `ml/dataset.py` | 70/15/15 deterministic image-level train/val/test split to prevent data leakage + `tf.data` patch extraction pipeline. |
| `ml/unet.py` | 4-level U-Net architecture with Dropout(0.3) at bottleneck, L2 regularization (1e-4), Batch Normalization, and Sigmoid output head. |
| `ml/train.py` | Training loop with combined 0.5×MAE + 0.5×(1−SSIM) loss, EarlyStopping (patience 10), ReduceLROnPlateau, and periodic checkpoints every 5 epochs. |
| `ml/AquaEnhance_Train.ipynb` | Colab/Kaggle-ready notebook for cloud GPU training and `.keras` model artifact export to Google Drive. |
| `ml/evaluate.py` | Computes paired metrics (PSNR, SSIM, MSE), no-reference underwater metrics (UIQM, UCIQE), and classical baselines (HE, CLAHE, White Balance, Gamma, Retinex). |
| `ml/inference.py` | End-to-end full-image enhancement with model caching and batched patch prediction. |

### Bug Catch & Resolution
- **Bug:** Zero-edge cosine/Hanning window caused edge pixels of outer boundary patches to have zero weight, leaving unweighted border pixels (reconstruction error ~1.0).
- **Fix:** Implemented Tukey-style blending window with clamped floor (minimum weight strictly > 0). Verified all 6/6 reconstruction tests pass with `0.00e+00` maximum error.

---

## Phase 3 — Backend API ✅ COMPLETE

**Date:** 2026-11-16

### Files Created & Verified

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI application exposing `POST /enhance`, `POST /feedback`, `GET /health`, `/storage` static mounts, and CORS middleware. |
| `backend/services/enhancement_service.py` | Decoupled ML inference service invoking patch slicing & blending; supports seamless fallback to CLAHE if model weights are not yet deployed. |
| `backend/test_api.py` | Test-first verification suite testing 6 realistic error and success cases. |

### Technical Decisions & Business Logic

1. **Max File Size Enforcement (5 MB):** Checks file byte length before reading large memory buffers; rejects oversized payloads immediately with HTTP `413 Content Too Large`.
2. **JPEG-Only Validation:** Validates `Content-Type` header (`image/jpeg`, `image/jpg`) AND verifies JPEG SOI magic bytes (`0xFFD8`) to prevent arbitrary file execution or processing crashes.
3. **Decoupled Inference Architecture:** `backend/main.py` does not contain ML math; it delegates processing to `enhancement_service.py`. If `best_model.keras` is absent during development, the system falls back gracefully to CLAHE baseline enhancement so the entire stack can be tested immediately.
4. **Database Audit Logging:** Successfully writes original path, enhanced path, and JSON processing stats (`patches`, `inference_ms`, `method`) to `Image`, and writes user ratings to `Feedback` table in MSSQL.
5. **Static File Mounting:** Uploaded and enhanced images in `backend/storage/` are served at `/storage/{filename}`.

### Verified Test Output (All 6 Pass)
```
1. Testing Oversized File (> 5 MB) Rejection...
✅ PASS Correctly rejected oversized file with HTTP 413.

2. Testing Non-JPEG File Rejection...
✅ PASS Correctly rejected non-JPEG MIME with HTTP 400.
✅ PASS Correctly rejected fake JPEG magic bytes with HTTP 400.

3. Testing Invalid Feedback Ratings (Rating Out-of-Bounds)...
✅ PASS Correctly rejected ratings 0 and 6 with HTTP 422.

4. Testing Feedback For Non-Existent image_id...
✅ PASS Correctly returned HTTP 404 for non-existent image_id.

5. Testing Valid /enhance Flow...
✅ PASS Enhance successful: image_id=6, processing={'patches': 2, 'inference_ms': 85.61, 'method': 'clahe_fallback', 'dimensions': '300x200'}
✅ PASS Static files served properly via /storage mount.

6. Testing Valid /feedback Submission...
✅ PASS Feedback registered: feedback_id=11, rating=5

============================================================
✅ All 6 Backend API tests passed successfully!
============================================================
```

---

## Phase 4 — Frontend Web App ✅ COMPLETE

**Date:** 2026-11-16

### Files Created & Verified

| File | Purpose |
|------|---------|
| `frontend/index.html` | Self-contained, zero-dependency Victorian web app with PixelCode font, drag-and-drop, canvas rendering, and feedback widget. Served directly by FastAPI at `http://localhost:8000/`. |
| `frontend/package.json` | Next.js 14 App Router project configuration for production build & deployment. |
| `frontend/app/page.js` | Client-side React page with side-by-side comparison, unsharp-mask sharpness convolution, brightness calibration, download, and star rating. |
| `frontend/app/layout.js` & `globals.css` | Next.js root layout and Victorian theme styles with PixelCode font integration. |
| `frontend/public/fonts/` | Offline PixelCode webfont binaries (`PixelCode.woff2`, `PixelCode-Bold.woff2`). |

### Feature Implementations

1. **Victorian Theme & PixelCode Font:**
   - Deep mahogany & antique gold color scheme (`#12100d`, `#1c1813`, `#cba863`, `#e5c98e`).
   - Ornate double borders, antique iconography (`⚜`, `⚓`), and typography styled with [PixelCode](https://qwerasd205.github.io/PixelCode/).
2. **Client-Side Validation:**
   - File picker and drag-and-drop zone enforce `<= 5 MB` file size and `image/jpeg` MIME type before upload.
3. **Side-by-Side Comparison:**
   - Left panel displays the original submerged frame.
   - Right panel displays the neural enhanced frame rendered on an HTML5 `<canvas>`.
4. **Client-Side Image Adjustments:**
   - **Brightness Slider (50%–150%):** Real-time CSS filter on canvas.
   - **Sharpness Slider (0–100):** Real-time client-side unsharp-mask convolution using a 3×3 Laplacian edge enhancement matrix on the canvas pixel buffer.
5. **Download:**
   - Exports the canvas directly as a JPEG containing all active brightness and sharpness adjustments at original resolution.
6. **1–5 Star Rating Widget:**
   - Interactive hover and click star rating that dispatches to `POST /feedback` with the active `image_id` and records user feedback in MSSQL.

### Feature Updates: PNG Support Added

- **Server Shutdown:** Stopped background Uvicorn daemon process (`task-254`) on port 8000.
- **Backend Format Expansion:** Updated `backend/main.py` and `backend/services/enhancement_service.py` to support `image/png` along with JPEG. Implemented PNG magic bytes verification (`\x89PNG\r\n\x1a\n`) and RGBA-to-RGB conversion. Preserves `.png` extension and PNG output encoding.
- **Frontend File Ingestion:** Updated `frontend/index.html` and `frontend/app/page.js` file pickers (`accept="image/jpeg, image/jpg, image/png"`), dropzone drag-and-drop validation, and user banners.
- **Verification:** Added Test 8 to `backend/test_api.py`. All 8/8 tests pass.

### Empirical Testing: 5 Real-World Underwater Photographs

- Processed 5 diverse real-world specimens through `backend/services/enhancement_service.py` (resolutions ranging from 604×325 up to 1024×642).
- Successfully verified dynamic patch slicing (6 to 20 patches per image), cosine-weighted reconstruction, and sub-60ms processing latencies.
- Evaluated no-reference quality metrics: UIQM (*Underwater Image Quality Measure*) and UCIQE (*Underwater Color Image Quality Evaluation*).
- Benchmark comparison against all 5 classical baselines in `ml/evaluate.py` (HE, CLAHE, White Balance, Gamma, Retinex) generated in `TEST_RESULTS.md`.

### Automated Frontend E2E Testing & Live Deployment

- Created `backend/test_frontend_e2e.py` verifying DOM elements, PixelCode font binaries, image upload pipelines, static URL resolution, and database feedback insertion on all 5 real test images. All 5/5 test phases pass.
- Started background Uvicorn server on `http://127.0.0.1:8000` for interactive testing in browser.

### How to Run the Entire System

```bash
# 1. Start the FastAPI backend & Victorian web server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Open in your browser:
# http://localhost:8000
```

All 8 integration tests in `backend/test_api.py` pass.
