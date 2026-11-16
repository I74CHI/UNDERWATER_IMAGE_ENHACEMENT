"""
train.py — Training script for AquaEnhance U-Net.

Designed to run on Google Colab or Kaggle with a GPU.
If running locally, TensorFlow must detect your GPU.

Outputs (all written to ml/model/):
    best_model.keras              — lowest val_loss weights (use this for deployment)
    ckpt_epoch_005.keras, etc.    — periodic checkpoint every 5 epochs

Usage:
    python ml/train.py

Requires TF >= 2.12 for @register_keras_serializable.
"""

import sys
from pathlib import Path

import tensorflow as tf

# Make sibling modules importable when called from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_pairs, split_pairs, build_dataset
from unet    import build_unet

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)
BEST_PATH = str(MODEL_DIR / "best_model.keras")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 8
MAX_EPOCHS = 100
LR         = 1e-4
CKPT_EVERY = 5    # save a checkpoint every N epochs


# ── Custom loss ───────────────────────────────────────────────────────────────
# @register_keras_serializable lets Keras save/load this function inside
# the .keras file — no need to pass custom_objects= when loading.
@tf.keras.saving.register_keras_serializable(package="AquaEnhance")
def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    0.5 × MAE  +  0.5 × (1 − SSIM)

    MAE  : pushes each pixel toward the correct value.
    SSIM : preserves edges, textures, and local contrast.
    Combining both balances pixel-level accuracy with structural quality.
    """
    mae  = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.5 * mae + 0.5 * (1.0 - ssim)


# ── Periodic checkpoint callback ──────────────────────────────────────────────
class CheckpointEveryN(tf.keras.callbacks.Callback):
    """Save the model every `every` epochs regardless of val_loss."""

    def __init__(self, save_dir: Path, every: int = 5):
        super().__init__()
        self.save_dir = save_dir
        self.every    = every

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.every == 0:
            path = str(self.save_dir / f"ckpt_epoch_{epoch + 1:03d}.keras")
            self.model.save(path)
            print(f"\n  ↳ Checkpoint → {path}")


# ── Main training function ────────────────────────────────────────────────────
def train():
    # Allow TF to grow GPU memory on demand instead of grabbing it all at once
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    # Build datasets (test split is evaluated separately via evaluate.py)
    pairs = get_pairs()
    train_pairs, val_pairs, _ = split_pairs(pairs)

    train_ds = build_dataset(train_pairs, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = build_dataset(val_pairs,   batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=combined_loss,
        metrics=["mae"],
    )
    model.summary(line_length=90)

    # Callbacks
    callbacks = [
        # ① Save best model (by val_loss) → used for deployment
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # ② Save a checkpoint every 5 epochs for recovery
        CheckpointEveryN(MODEL_DIR, every=CKPT_EVERY),
        # ③ Stop early if val_loss stops improving for 10 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # ④ Halve the learning rate when val_loss plateaus for 5 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
    )

    print(f"\nTraining complete.  Best model → {BEST_PATH}")
    return history


if __name__ == "__main__":
    train()
