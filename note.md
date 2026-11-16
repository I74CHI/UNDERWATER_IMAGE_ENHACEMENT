# AquaEnhance — Project Guide and Workflow

**Project:** AquaEnhance, an underwater image enhancement web application  
**Project type:** B.Tech final-year project  
**Last reviewed:** 2026-11-16

This note describes the repository as it exists now: its components, the upload-to-result workflow, the enhancement and comparison methods, development tools, setup, testing, and known limitations. It does not contain database passwords, API keys, or other `.env` values.

## 1. Project purpose

Underwater photographs often have color casts, reduced contrast, and lost detail because water absorbs and scatters light. AquaEnhance accepts a JPG, JPEG, or PNG image, creates an enhanced version, and presents it alongside the original. It also compares the result with five classical image-enhancement methods:

1. Histogram Equalization
2. CLAHE
3. White Balance
4. Gamma Correction
5. Retinex-based Enhancement

The interface also offers brightness and sharpness controls, image download, and a 1–5 star rating. Feedback is saved in the database; it does **not** trigger online retraining.

## 2. System architecture

```text
Browser
  ├─ FastAPI-served frontend: GET /
  │    └─ frontend\index.html (standalone HTML, CSS, and JavaScript)
  └─ Optional Next.js frontend: frontend\app\
       └─ Sends API requests to FastAPI

FastAPI (backend\main.py)
  ├─ validates uploads and ratings
  ├─ calls backend\services\enhancement_service.py
  ├─ stores original, enhanced, and baseline images in backend\storage\
  ├─ reads/writes image and feedback records in the database
  └─ serves saved images under /storage/

ML modules (ml\)
  ├─ overlapping image patches
  ├─ U-Net inference when a trained model file is installed
  ├─ optical-restoration fallback when it is not
  ├─ five classical comparison methods
  └─ image-quality metrics
```

The project has two frontend entry points:

- The standalone `frontend\index.html` page is served by FastAPI at `http://127.0.0.1:8000/`. It does not need a frontend build step.
- The Next.js 14 App Router application is in `frontend\app\` and runs separately on port 3000. Its API URL is configured with `NEXT_PUBLIC_API_URL`.

The FastAPI-served HTML is the page used when opening port 8000 directly. The Next.js app is a separate frontend option, not the page FastAPI serves at `/`.

## 3. Repository map

| Location | Responsibility |
|---|---|
| `backend\main.py` | FastAPI app, routes, upload validation, storage URLs, CORS, and database operations |
| `backend\services\enhancement_service.py` | Enhancement/fallback selection, baseline image generation, metrics, and processing metadata |
| `backend\storage\` | Uploaded images, enhanced images, and generated baseline results |
| `backend\test_api.py` | Backend endpoint and comparison-response tests |
| `backend\test_frontend_e2e.py` | Frontend/API integration test; currently contains machine-specific image paths |
| `database\db.py` | DB-API connection helper for SQL Server or MySQL |
| `database\schema.sql` | Microsoft SQL Server schema |
| `database\schema_mysql.sql` | MySQL schema for the Docker fallback |
| `database\test_db.py` | Database connection, constraints, and relationship tests |
| `frontend\index.html` | Standalone frontend served by FastAPI |
| `frontend\app\page.js` | Next.js client page |
| `frontend\app\layout.js` | Next.js root layout and metadata |
| `frontend\app\globals.css` | Next.js global theme and comparison-card styles |
| `frontend\package.json` | Next.js, React, and npm scripts |
| `frontend\public\fonts\` | Local PixelCode fonts |
| `ml\patches.py` | Patch extraction and overlap-weighted reconstruction used by training and inference |
| `ml\patch_utils.py` | A second patch utility implementation used by `ml\test_patches.py` |
| `ml\unet.py` | TensorFlow/Keras U-Net model |
| `ml\inference.py` | Cached model loading, batched inference, and patch reconstruction |
| `ml\train.py` | Model training loop and checkpoints |
| `ml\dataset.py` | Paired-image loading, train/validation/test split, and `tf.data` pipeline |
| `ml\prepare_dataset.py` | Utility to split a UIEB folder/archive into train/validation/test folders |
| `ml\data\download_uieb.py` | Kaggle CLI-based UIEB downloader |
| `ml\evaluate.py` | Paired and no-reference metrics, classical baselines, and optical fallback |
| `ml\AquaEnhance_Train.ipynb` | Colab-oriented training notebook |
| `docker-compose.yml` | Optional MySQL 8.3 service with persistent storage |
| `PROJECT_LOG.md`, `RULES.md` | Historical project log and project requirements/decisions |

## 4. Image upload and processing workflow

1. **Choose an image.** The browser accepts JPG/JPEG/PNG files up to 5 MB. The standalone page also supports drag-and-drop.
2. **Submit the upload.** The frontend sends multipart form data containing a `file` field to `POST /enhance`.
3. **Validate the upload.** FastAPI checks the declared MIME type, enforces the 5 MB limit, and checks JPEG or PNG magic bytes.
4. **Save the original.** A short unique identifier is generated. The input is saved under a name such as `orig_<id>.jpg` or `orig_<id>.png`.
5. **Enhance and compare.** The service converts the image to RGB, normalizes pixels to `[0, 1]`, chooses the U-Net or fallback path, and generates the five classical baseline results.
6. **Save results.** The enhanced image is saved as `enh_<id>.<ext>`. Baselines are saved as `base_hist_eq_<id>.<ext>`, `base_clahe_<id>.<ext>`, `base_white_balance_<id>.<ext>`, `base_gamma_<id>.<ext>`, and `base_retinex_<id>.<ext>`.
7. **Calculate metrics.** UIQM and UCIQE are computed for AquaEnhance and each baseline.
8. **Record processing.** The database receives the original filename, enhanced filename, and JSON processing information. Baseline filenames and their metrics are also present in the processing metadata.
9. **Return data and display it.** The response includes the image ID, original/enhanced URLs, processing details, and six comparison entries (AquaEnhance plus five baselines). The frontend displays their images and metrics.
10. **Optional user actions.** Brightness is adjusted with a canvas filter and sharpness with a 3×3 Laplacian-based edge operation. The download exports the adjusted canvas. A rating is submitted separately to `POST /feedback`.

Image files are kept on disk; the database stores paths and metadata rather than image BLOBs.
The `processing_info.inference_ms` field is measured across the complete service operation, including baseline generation and metric calculation, so it is broader than model-only inference time.

### Upload and feedback limits

- Maximum upload size: 5 MiB (5 × 1024 × 1024 bytes).
- Accepted formats: JPEG and PNG.
- Feedback: integer rating from 1 through 5, associated with a valid image ID.
- The service reads the uploaded body before checking its byte length; this is a size limit, not a streaming upload limit.

## 5. Enhancement paths

### U-Net path

The service looks for `ml\model\best_model.keras`. If found, it tries `ml\inference.py`; if model loading or prediction raises an exception, the service falls back to optical restoration. The inference module:

1. Loads and caches the Keras model for reuse.
2. Converts an input image to RGB float values in `[0, 1]`.
3. Slices it into 256 × 256 patches, using a 32-pixel overlap (stride 224).
4. Predicts patches in batches (default batch size 4).
5. Reconstructs the full image with weighted overlap blending and crops padding.

The U-Net has four encoder levels (64, 128, 256, 512 filters), a 1024-filter bottleneck with dropout, and a four-level decoder with skip connections. Convolution layers use batch normalization and L2 regularization. The output is RGB in `[0, 1]`.

### Current fallback path

No `ml\model\best_model.keras` file was present when this guide was reviewed. In that case the backend uses `land_like_restore` from `ml\evaluate.py` and reports the method as `land_like_optical`. This fallback:

1. Compensates for red (and, when appropriate, blue) channel attenuation.
2. Applies gray-world color balancing.
3. Stretches each channel using the 0.8th and 99.2nd percentiles.
4. Applies CLAHE to the CIELAB lightness channel.
5. Boosts HSV saturation for vibrance.

This is a working optical enhancement path, but it is **not U-Net inference**. To use the learned model, train/export `best_model.keras` and place it at the path above.

## 6. Comparison algorithms and quality scores

| Display name | Implementation |
|---|---|
| Histogram Equalization | Independently equalizes the three RGB channels using OpenCV |
| CLAHE | Applies Contrast Limited Adaptive Histogram Equalization to the LAB lightness channel |
| White Balance | Gray-world correction using per-channel image means |
| Gamma Correction | Power-law correction with default gamma 1.5 to brighten dark regions |
| Retinex-based Enhancement | Single-scale Retinex using log intensity minus log Gaussian illumination, normalized to `[0, 1]` |

The browser shows **UIQM** and **UCIQE**:

- **UIQM** (Underwater Image Quality Measure) combines colorfulness, sharpness, and contrast components.
- **UCIQE** (Underwater Color Image Quality Evaluation) uses Lab chroma variation, lightness contrast, and HSV saturation.
- These are no-reference metrics: they do not require a ground-truth enhanced image. Higher scores are commonly treated as better, but they are indicators, not a guarantee of perceptual quality.

`ml\evaluate.py` also implements paired **PSNR**, **SSIM**, and **MSE**. Those need an aligned reference image; the live upload API does not calculate them because an uploaded image has no supplied ground-truth reference.

The comparison metrics describe the generated server-side images. They do not change when the user adjusts the browser's brightness or sharpness sliders.

## 7. API reference

### `GET /`

Serves `frontend\index.html` when that file exists.

### `GET /health`

Returns API status and a database connection status. The API status can be `ok` while the database field reports an error.

### `POST /enhance`

Multipart request:

```text
file: <JPEG or PNG bytes>
```

Successful response shape:

```json
{
  "image_id": 123,
  "original_url": "http://localhost:8000/storage/orig_<id>.jpg",
  "enhanced_url": "http://localhost:8000/storage/enh_<id>.jpg",
  "processing_info": {
    "patches": 4,
    "inference_ms": 120.5,
    "method": "land_like_optical",
    "dimensions": "640x480"
  },
  "comparisons": [
    {
      "method": "AquaEnhance (Optical)",
      "image_url": "http://localhost:8000/storage/enh_<id>.jpg",
      "metrics": { "uiqm": 0.7, "uciqe": 9.1 }
    }
  ]
}
```

The real `comparisons` array contains AquaEnhance and all five classical methods. Error cases include invalid format (400), oversized file (413), processing/database errors (500).

### `POST /feedback`

JSON request:

```json
{ "image_id": 123, "rating": 5 }
```

The backend validates the rating, checks the image row exists, and inserts a feedback row. Invalid request fields produce 422; unknown image IDs produce 404.

### `GET /storage/{filename}`

Serves saved original, enhanced, or baseline files from the configured storage directory.

## 8. Database model

The SQL Server schema creates:

- **Image** — integer primary key, original and enhanced filenames, upload timestamp, and JSON processing information.
- **Feedback** — integer primary key, image foreign key, rating constrained to 1–5, and timestamp. Deleting an image cascades to its feedback.

`database\db.py` selects the driver using `DB_DRIVER`:

- `mssql`: `pyodbc` and ODBC Driver 17 for SQL Server.
- `mysql`: `mysql-connector-python`.

The SQL Server schema is in `database\schema.sql`; the MySQL 8.x alternative is in `database\schema_mysql.sql`. `docker-compose.yml` starts the MySQL fallback and expects `DB_USER` and `DB_PASSWORD` in the Compose environment. Do not put real credentials in documentation or commit them to source control.

## 9. Model training and dataset workflow

### Dataset source

The intended supervised dataset is UIEB: 890 paired raw/reference underwater images. `ml\data\download_uieb.py` calls the Kaggle CLI and expects Kaggle credentials to be configured outside the repository. Never commit `kaggle.json`.

### Dataset loader

`ml\dataset.py` matches raw/reference filenames by stem, uses a deterministic seed (42), and splits matched images into 70% training, 15% validation, and 15% test **before** extracting patches. This avoids placing patches from one photo in both training and validation sets. It then builds TensorFlow `tf.data` datasets of paired normalized patches.

### Training

`ml\train.py` builds and trains the U-Net with:

- Adam optimizer, learning rate `1e-4`
- Batch size 8, up to 100 epochs
- Loss: `0.5 × MAE + 0.5 × (1 − SSIM)`
- Best-model saving based on validation loss
- A checkpoint every 5 epochs
- Early stopping (patience 10) and learning-rate reduction on plateaus (patience 5)

The intended deployment model is `ml\model\best_model.keras`. The notebook `ml\AquaEnhance_Train.ipynb` is designed for Google Colab, GPU training, Kaggle download, and Google Drive persistence.
The standalone `ml\train.py` script uses the train and validation subsets but does not run evaluation on the held-out test subset; the notebook includes a separate test-set evaluation step.

### Important dataset-layout mismatch

There are currently two different data layouts in the repository:

- `ml\dataset.py` and the Kaggle downloader expect `ml\data\raw-890\` and `ml\data\reference-890\`, then split them in code.
- `ml\prepare_dataset.py` writes `ml\data\train\raw\`, `ml\data\train\reference\`, and corresponding `val`/`test` directories.
- `ml\train.py` calls `dataset.get_pairs()` with its default `raw-890` and `reference-890` paths.

Therefore, the output of `prepare_dataset.py` is not consumed by the current default training loader without path/code changes. Align the paths or choose one workflow before training. At review time, neither UIEB data folder nor a trained model artifact was present in the repository.

## 10. Tools and technologies

| Tool/technology | Role |
|---|---|
| Python | Backend, ML pipeline, evaluation, and test scripts |
| FastAPI + Uvicorn | HTTP API and local app server |
| NumPy | Image arrays, normalization, and numerical processing |
| Pillow | Image decoding/encoding and RGB conversion |
| OpenCV | Classical enhancement methods and color-space operations |
| scikit-image | SSIM and PSNR metric functions |
| TensorFlow/Keras | U-Net model definition, training, saving, and inference |
| Microsoft SQL Server + pyodbc | Primary database option |
| MySQL + mysql-connector-python | Alternative database connector |
| Docker Compose | Optional local MySQL service |
| React + Next.js 14 | Optional separately served frontend |
| HTML/CSS/JavaScript | Standalone frontend, canvas rendering, and controls |
| Kaggle CLI | Downloading UIEB data |
| Google Colab / Google Drive | Suggested GPU training and model persistence |
| VS Code | Project editing and local development environment |

The root `requirements.txt` lists Python packages used by the API, image pipeline, U-Net workflow, both database options, tests, and UIEB notebook utilities. A pip dry-run resolved the manifest in the current Python environment without installing packages. It does not pin package versions, so it is not a lockfile; select versions compatible with the target Python and especially with TensorFlow. The frontend manifest declares Next.js `^14.2.0`, React `^18.2.0`, and React DOM `^18.2.0`; there is no npm lockfile in the project inventory.

## 11. Setup and run

### Backend and standalone frontend

1. Install a Python version compatible with the selected packages and configure the database variables in `backend\.env` (do not publish that file).
2. Create the database using `database\schema.sql` for SQL Server or `database\schema_mysql.sql` for MySQL.
3. Install the Python packages:

```powershell
python -m pip install -r requirements.txt
```

The file includes both database connectors and ML/notebook tools. SQL Server also requires the system ODBC Driver 17. TensorFlow must have a wheel compatible with the selected Python/platform; training can be run in the Colab notebook when local compatibility or GPU resources are an issue.
4. From the repository root, start the app:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

5. Open `http://127.0.0.1:8000/`. Check `http://127.0.0.1:8000/health` for the API and database status.

The backend loads key/value settings from `backend\.env` if present, using `setdefault` so values already present in the process environment take precedence. Relevant setting names include `DB_DRIVER`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `STORAGE_DIR`, `API_BASE_URL`, and `CORS_ORIGINS`.

### Optional Next.js frontend

Run the API on port 8000 in one terminal. In a second PowerShell terminal:

```powershell
Set-Location frontend
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
npm install
npm run dev
```

Then open `http://localhost:3000/`. The backend `CORS_ORIGINS` must include the Next.js page origin, normally `http://localhost:3000`.

### Optional MySQL container

Supply `DB_USER` and `DB_PASSWORD` to Docker Compose using a local, untracked environment file or environment variables, then run:

```powershell
docker compose up -d
```

Configure the backend with `DB_DRIVER=mysql`, host `localhost`, and port `3306`. SQL Server is the primary documented option.

### Optional UIEB download and training

1. Install/configure the Kaggle CLI and its credential file outside the repository.
2. Run `python ml\data\download_uieb.py` and verify the expected `raw-890`/`reference-890` folders exist.
3. Resolve the dataset-layout mismatch described above before training.
4. In a TensorFlow-compatible environment (the notebook metadata targets Python 3.10), run `python ml\train.py` or follow `ml\AquaEnhance_Train.ipynb` in Colab.
5. Place the exported best model at `ml\model\best_model.keras` for backend U-Net inference.

## 12. Tests and verification

Run from the repository root:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python ml\test_patch.py
python ml\test_patches.py
python backend\test_api.py
python database\test_db.py
```

- `ml\test_patch.py` demonstrates the missing-border bug in a naïve patch implementation, then checks round-trip reconstruction at six image sizes. The observed current result was 6/6 passing with zero reported maximum error.
- `ml\test_patches.py` checks the separate `ml\patch_utils.py` implementation.
- `backend\test_api.py` checks upload rejection/success, all six comparison results and their images/metrics, feedback, root-page serving, fonts, and PNG handling. The full API suite passed in this workspace.
- `database\test_db.py` requires a reachable test database and checks inserts, the rating constraint, and the image-feedback foreign key.
- `backend\test_frontend_e2e.py` contains hard-coded machine-specific image paths and requires a working database. Update `TEST_IMAGES` before using it on another machine.

The browser upload flow was also verified after the asset-URL fix: the loading indicator cleared and the six comparison images loaded. A Next.js production build was not verified in this environment because `npm` was unavailable.

## 13. Current limitations and maintenance notes

1. **Model/data are not included.** The current checkout had no `ml\model\best_model.keras`, UIEB `raw-890` folder, or UIEB `reference-890` folder. Without the model, uploads use the optical fallback.
2. **Python dependencies are not pinned.** `requirements.txt` provides the dependency list, but versions should be tested and pinned for reproducible deployment.
3. **Dataset layouts disagree.** Align `prepare_dataset.py` and `dataset.py` before using the split-folder preparation output for training.
4. **Patch helpers are duplicated.** Runtime training/inference uses `ml\patches.py`; the separate `ml\test_patches.py` test imports `ml\patch_utils.py`. Keep their behavior aligned or consolidate them deliberately.
5. **Training is not live.** Collected user ratings are database records only.
6. **Storage grows over time.** Each successful upload creates the original, enhanced image, and five baseline images. Define a retention/cleanup policy for long-running deployments.
7. **No authentication is implemented.** The app is a research/demo project; do not expose its upload and feedback endpoints publicly without appropriate deployment protections.
8. **UI progress text is generic.** The page says it is running neural inference even when the backend is using its optical fallback; inspect `processing_info.method` to determine the actual method.
9. **Image URLs must use the API host.** Both frontends normalize returned image URLs to the configured API origin. For the Next.js app, set `NEXT_PUBLIC_API_URL` to the reachable backend origin; otherwise image loading and canvas rendering can fail.
10. **Both frontend implementations need upkeep.** The standalone HTML page and the Next.js page contain separate UI logic and should be kept behaviorally consistent.

