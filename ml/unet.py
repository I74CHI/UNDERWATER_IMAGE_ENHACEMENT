"""
unet.py — Standard U-Net for underwater image-to-image enhancement.

Architecture
------------
  Input   : (256, 256, 3)   float32 RGB in [0, 1]
  Encoder : 4 blocks — double Conv2D + MaxPool2D
            filter sizes: 64 → 128 → 256 → 512
  Neck    : double Conv2D (1024 filters) + Dropout(0.3)
  Decoder : 4 blocks — Conv2DTranspose + concat(skip) + double Conv2D
  Output  : Conv2D(3, 1×1, sigmoid) → enhanced RGB in [0, 1]

Regularisation: L2(1e-4) on every Conv2D kernel weight.
Normalisation : BatchNormalization after every Conv2D (use_bias=False
                because BN already handles the bias term).

Usage
-----
    from unet import build_unet
    model = build_unet()
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

L2 = 1e-4   # L2 regularisation factor


def _double_conv(x: tf.Tensor, filters: int) -> tf.Tensor:
    """
    The U-Net building block:
      Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU

    Two convolutions with the same filter count keep the spatial size
    unchanged (padding="same") while learning richer features.
    """
    reg = regularizers.l2(L2)
    for _ in range(2):
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same",
            use_bias=False, kernel_regularizer=reg
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(input_shape: tuple = (256, 256, 3)) -> Model:
    """
    Build and return the AquaEnhance U-Net model.

    Args
    ----
    input_shape : (H, W, C) of one patch — default (256, 256, 3).

    Returns
    -------
    tf.keras.Model with named input "raw_patch" and output "enhanced_patch".
    """
    reg = regularizers.l2(L2)
    inp = layers.Input(shape=input_shape, name="raw_patch")

    # ── Encoder (feature extraction + downsampling) ───────────────────────────
    e1 = _double_conv(inp, 64)
    e2 = _double_conv(layers.MaxPool2D(2)(e1), 128)
    e3 = _double_conv(layers.MaxPool2D(2)(e2), 256)
    e4 = _double_conv(layers.MaxPool2D(2)(e3), 512)

    # ── Bottleneck (deepest representation) ───────────────────────────────────
    neck = _double_conv(layers.MaxPool2D(2)(e4), 1024)
    neck = layers.Dropout(0.3)(neck)   # only here — moderate overfitting control

    # ── Decoder (feature synthesis + upsampling) ──────────────────────────────
    # Each level: upsample → concat skip connection → double conv
    up4 = layers.Conv2DTranspose(512, 2, strides=2, padding="same", kernel_regularizer=reg)(neck)
    d4  = _double_conv(layers.Concatenate()([up4, e4]), 512)

    up3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same", kernel_regularizer=reg)(d4)
    d3  = _double_conv(layers.Concatenate()([up3, e3]), 256)

    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same", kernel_regularizer=reg)(d3)
    d2  = _double_conv(layers.Concatenate()([up2, e2]), 128)

    up1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same", kernel_regularizer=reg)(d2)
    d1  = _double_conv(layers.Concatenate()([up1, e1]), 64)

    # ── Output head ───────────────────────────────────────────────────────────
    # 1×1 convolution maps 64 feature maps → 3 RGB channels.
    # sigmoid keeps output in [0, 1] to match normalised input.
    out = layers.Conv2D(3, 1, activation="sigmoid", name="enhanced_patch")(d1)

    return Model(inp, out, name="AquaEnhance_UNet")


if __name__ == "__main__":
    model = build_unet()
    model.summary(line_length=90)
    print(f"\nTotal trainable parameters: {model.count_params():,}")
