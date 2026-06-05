# utils/mlp_utils.py

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import r2_score, mean_squared_error



# =========================
# Reproducibility
# =========================

RANDOM_SEED = 2026

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =========================
# Config
# =========================

@dataclass
class MLPTrainConfig:
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 128
    seed: int = RANDOM_SEED
    verbose: int = 0


# =========================
# Naming helpers
# =========================

def hidden_sizes_to_tag(hidden_sizes: Tuple[int, ...]) -> str:
    """
    (10,) -> h10
    (10,10) -> h10_10
    """
    return "h" + "_".join(str(h) for h in hidden_sizes)


# =========================
# Data helpers
# =========================

def ensure_2d_targets(y: np.ndarray) -> np.ndarray:
    """
    Ensures y has shape (N, 1)
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1).astype(np.float32)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.astype(np.float32)
    raise ValueError(f"Invalid y shape: {y.shape}")


# =========================
# Model
# =========================

def build_trust_mlp(
    input_dim: int,
    hidden_sizes: List[int],
    output_dim: int = 1,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    TRUST-compatible MLP:
    - Dense + ReLU hidden layers
    - Linear output
    - Explicit layer names
    """
    inputs = layers.Input(shape=(input_dim,), name="mlp_input")
    x = inputs

    for i, units in enumerate(hidden_sizes, start=1):
        x = layers.Dense(
            units,
            activation="relu",
            name=f"hidden_l{i}",
        )(x)

    outputs = layers.Dense(
        output_dim,
        activation="linear",
        name="rul_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="trust_mlp")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =========================
# Export helpers
# =========================

def export_layer_weights_dict(
    model: tf.keras.Model,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns:
        weights[layer_name] = {"W": W, "b": b}
    """
    weights = {}
    for layer in model.layers:
        params = layer.get_weights()
        if len(params) > 0:
            W, b = params
            weights[layer.name] = {"W": W, "b": b}
    return weights


def compute_final_metrics(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """
    Computes final regression metrics on train and validation sets.
    Returns MSE, RMSE, MAE, and R2.
    """
    # Predictions
    y_train_pred = model.predict(X_train, verbose=0).reshape(-1)
    y_val_pred = model.predict(X_val, verbose=0).reshape(-1)

    y_train_true = y_train.reshape(-1)
    y_val_true = y_val.reshape(-1)

    # Train metrics
    train_mse = mean_squared_error(y_train_true, y_train_pred)
    val_mse = mean_squared_error(y_val_true, y_val_pred)

    metrics = {
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "train_rmse": float(np.sqrt(train_mse)),
        "val_rmse": float(np.sqrt(val_mse)),
        "train_mae": float(np.mean(np.abs(y_train_true - y_train_pred))),
        "val_mae": float(np.mean(np.abs(y_val_true - y_val_pred))),
        "train_r2": float(r2_score(y_train_true, y_train_pred)),
        "val_r2": float(r2_score(y_val_true, y_val_pred)),
    }

    return metrics
