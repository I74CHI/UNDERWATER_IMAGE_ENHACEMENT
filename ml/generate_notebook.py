"""
generate_notebook.py — Generates AquaEnhance_Train.ipynb for Google Colab.

Run once:
    python ml/generate_notebook.py

The notebook is self-contained: it installs dependencies, configures
Kaggle credentials, downloads UIEB, and trains the U-Net end-to-end.
"""

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


CELLS = [
    md(
        "# 🌊 AquaEnhance — Underwater Image Enhancement\n\n"
        "**Google Colab Training Notebook**  \n"
        "Trains a U-Net on the UIEB dataset to enhance degraded underwater images.\n\n"
        "---\n"
        "**Before running:**\n"
        "1. `Runtime → Change runtime type → T4 GPU`\n"
        "2. Upload `kaggle.json` when Cell 2 prompts you\n"
        "3. Run cells top-to-bottom — Google Drive keeps the model after session ends"
    ),

    code(
        "# ── Cell 1 : Install dependencies ───────────────────────────────────\n"
        "!pip install -q kaggle scikit-image opencv-python-headless Pillow\n"
        "import tensorflow as tf\n"
        "print('TF version:', tf.__version__)\n"
        "print('GPU:', tf.config.list_physical_devices('GPU'))"
    ),

    code(
        "# ── Cell 2 : Configure Kaggle API key ───────────────────────────────\n"
        "# 1. Go to https://www.kaggle.com → Account → API → Create New Token\n"
        "# 2. Upload the kaggle.json file when the dialog appears below\n"
        "import os\n"
        "from google.colab import files\n"
        "\n"
        "uploaded = files.upload()          # select your kaggle.json\n"
        "os.makedirs('/root/.kaggle', exist_ok=True)\n"
        "!mv kaggle.json /root/.kaggle/\n"
        "!chmod 600 /root/.kaggle/kaggle.json\n"
        "print('Kaggle API key ready.')"
    ),

    code(
        "# ── Cell 3 : Mount Google Drive (persistent model storage) ──────────\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "import os\n"
        "DRIVE_DIR = '/content/drive/MyDrive/AquaEnhance'\n"
        "MODEL_DIR = DRIVE_DIR + '/model'\n"
        "DATA_DIR  = DRIVE_DIR + '/data'\n"
        "for d in [MODEL_DIR, DATA_DIR]:\n"
        "    os.makedirs(d, exist_ok=True)\n"
        "print('Drive mounted. Model will be saved to:', MODEL_DIR)"
    ),

    code(
        "# ── Cell 4 : Upload project files ────────────────────────────────────\n"
        "# OPTION A — if you have the project on GitHub:\n"
        "#   !git clone https://github.com/YOUR_USERNAME/AquaEnhance.git /content/AquaEnhance\n"
        "#   %cd /content/AquaEnhance\n"
        "#\n"
        "# OPTION B — manual upload (zip the ml/ folder, upload here):\n"
        "#   from google.colab import files\n"
        "#   files.upload()          # upload ml.zip\n"
        "#   !unzip -q ml.zip -d /content/\n"
        "\n"
        "# After either option, add ml/ to Python path:\n"
        "import sys\n"
        "sys.path.insert(0, '/content/ml')     # adjust if cloned elsewhere\n"
        "print('sys.path set. Ready to import project modules.')"
    ),

    code(
        "# ── Cell 5 : Download UIEB dataset ───────────────────────────────────\n"
        "import subprocess, os\n"
        "result = subprocess.run(\n"
        "    ['kaggle', 'datasets', 'download',\n"
        "     '-d', 'larjeck/uieb-dataset-raw-and-reference',\n"
        "     '-p', DATA_DIR, '--unzip'],\n"
        "    capture_output=True, text=True\n"
        ")\n"
        "print(result.stdout or result.stderr)\n"
        "\n"
        "raw_n = len(os.listdir(DATA_DIR + '/raw-890'))\n"
        "ref_n = len(os.listdir(DATA_DIR + '/reference-890'))\n"
        "print(f'raw-890: {raw_n} files | reference-890: {ref_n} files')"
    ),

    code(
        "# ── Cell 6 : Build train / val / test datasets ───────────────────────\n"
        "from pathlib import Path\n"
        "import dataset as ds\n"
        "\n"
        "# Point the module to the Drive data directory\n"
        "ds.RAW_DIR = Path(DATA_DIR) / 'raw-890'\n"
        "ds.REF_DIR = Path(DATA_DIR) / 'reference-890'\n"
        "\n"
        "pairs = ds.get_pairs(ds.RAW_DIR, ds.REF_DIR)\n"
        "train_pairs, val_pairs, test_pairs = ds.split_pairs(pairs)\n"
        "\n"
        "BATCH    = 8\n"
        "train_ds = ds.build_dataset(train_pairs, batch_size=BATCH, shuffle=True)\n"
        "val_ds   = ds.build_dataset(val_pairs,   batch_size=BATCH)\n"
        "test_ds  = ds.build_dataset(test_pairs,  batch_size=BATCH)\n"
        "print('Datasets built.')"
    ),

    code(
        "# ── Cell 7 : Build U-Net model ────────────────────────────────────────\n"
        "from unet import build_unet\n"
        "import tensorflow as tf\n"
        "\n"
        "model = build_unet()\n"
        "model.summary(line_length=90)\n"
        "print(f'Total parameters: {model.count_params():,}')"
    ),

    code(
        "# ── Cell 8 : Define loss and compile ─────────────────────────────────\n"
        "@tf.keras.saving.register_keras_serializable(package='AquaEnhance')\n"
        "def combined_loss(y_true, y_pred):\n"
        "    mae  = tf.reduce_mean(tf.abs(y_true - y_pred))\n"
        "    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))\n"
        "    return 0.5 * mae + 0.5 * (1.0 - ssim)\n"
        "\n"
        "model.compile(\n"
        "    optimizer=tf.keras.optimizers.Adam(1e-4),\n"
        "    loss=combined_loss,\n"
        "    metrics=['mae'],\n"
        ")\n"
        "print('Model compiled.')"
    ),

    code(
        "# ── Cell 9 : Train ────────────────────────────────────────────────────\n"
        "BEST_PATH = MODEL_DIR + '/best_model.keras'\n"
        "\n"
        "class CheckpointEveryN(tf.keras.callbacks.Callback):\n"
        "    def on_epoch_end(self, epoch, logs=None):\n"
        "        if (epoch + 1) % 5 == 0:\n"
        "            p = MODEL_DIR + f'/ckpt_epoch_{epoch+1:03d}.keras'\n"
        "            self.model.save(p)\n"
        "            print(f'  Checkpoint → {p}')\n"
        "\n"
        "callbacks = [\n"
        "    tf.keras.callbacks.ModelCheckpoint(\n"
        "        BEST_PATH, monitor='val_loss', save_best_only=True, verbose=1),\n"
        "    CheckpointEveryN(),\n"
        "    tf.keras.callbacks.EarlyStopping(\n"
        "        monitor='val_loss', patience=10,\n"
        "        restore_best_weights=True, verbose=1),\n"
        "    tf.keras.callbacks.ReduceLROnPlateau(\n"
        "        monitor='val_loss', factor=0.5,\n"
        "        patience=5, min_lr=1e-7, verbose=1),\n"
        "]\n"
        "\n"
        "history = model.fit(\n"
        "    train_ds, validation_data=val_ds,\n"
        "    epochs=100, callbacks=callbacks,\n"
        ")\n"
        "print('Training complete. Best model at:', BEST_PATH)"
    ),

    code(
        "# ── Cell 10 : Plot training curves ───────────────────────────────────\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax1.plot(history.history['loss'],     label='train')\n"
        "ax1.plot(history.history['val_loss'], label='val')\n"
        "ax1.set(title='Loss (combined)', xlabel='Epoch'); ax1.legend()\n"
        "\n"
        "ax2.plot(history.history['mae'],     label='train')\n"
        "ax2.plot(history.history['val_mae'], label='val')\n"
        "ax2.set(title='MAE', xlabel='Epoch'); ax2.legend()\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(MODEL_DIR + '/training_curves.png', dpi=120)\n"
        "plt.show()"
    ),

    code(
        "# ── Cell 11 : Evaluate on test set ───────────────────────────────────\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "import evaluate as ev\n"
        "from inference import enhance\n"
        "\n"
        "all_metrics = []\n"
        "for raw_path, ref_path in test_pairs:\n"
        "    enhanced = enhance(raw_path, BEST_PATH)\n"
        "    ref      = np.array(Image.open(ref_path).convert('RGB'),\n"
        "                        dtype=np.float32) / 255.0\n"
        "    all_metrics.append(ev.compute_all_metrics(enhanced, reference=ref))\n"
        "\n"
        "print('\\nTest set averages')\n"
        "print('-' * 30)\n"
        "for key in all_metrics[0]:\n"
        "    vals = [m[key] for m in all_metrics]\n"
        "    print(f'  {key:8s}: {np.mean(vals):.4f}')"
    ),

    code(
        "# ── Cell 12 : Visual comparison (first 3 test images) ───────────────\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n"
        "import numpy as np\n"
        "\n"
        "fig, axes = plt.subplots(3, 2, figsize=(10, 12))\n"
        "for i, (raw_path, _) in enumerate(test_pairs[:3]):\n"
        "    raw = np.array(Image.open(raw_path).convert('RGB'),\n"
        "                   dtype=np.float32) / 255.0\n"
        "    enh = enhance(raw_path, BEST_PATH)\n"
        "    axes[i, 0].imshow(raw); axes[i, 0].set_title('Raw (Degraded)')\n"
        "    axes[i, 0].axis('off')\n"
        "    axes[i, 1].imshow(enh); axes[i, 1].set_title('Enhanced (U-Net)')\n"
        "    axes[i, 1].axis('off')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(MODEL_DIR + '/sample_results.png', dpi=150)\n"
        "plt.show()\n"
        "print('Saved to', MODEL_DIR + '/sample_results.png')"
    ),
]

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "cells": CELLS,
}

out = Path(__file__).parent / "AquaEnhance_Train.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(NOTEBOOK, f, indent=2, ensure_ascii=False)

print(f"Notebook written → {out}")
