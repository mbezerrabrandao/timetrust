from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras import optimizers

# NOTE:
# - TRUST constants should live in one place to avoid circular imports.
from public_time_trust.utils.milp_utils import TRUST_MODE_FULL, TRUST_MODE_SENSORS, TRUST_MODE_WINDOWS
from public_time_trust.utils.mlp_utils import RANDOM_SEED, build_trust_mlp


# =========================
# Weights extraction (Keras -> TRUST format)
# =========================

def extract_mlp_weights(results: Dict[str, Any], run_key: Optional[str] = None, verbose: bool = False):
    """
    Extract all hidden layer weights, biases, and output layer weights
    from a saved results dictionary.

    Expected structure:
        results[run_key]["hidden_sizes"]
        results[run_key]["weights"][layer_name]["W"|"b"]

    If run_key is None, `results` itself is assumed to be the entry.
    """
    if run_key is None:
        entry = results
    else:
        if run_key not in results:
            raise KeyError(f"run_key '{run_key}' not found in results")
        entry = results[run_key]

    if "weights" not in entry or "hidden_sizes" not in entry:
        raise KeyError("Missing 'weights' or 'hidden_sizes' in results entry")

    hidden_sizes = entry["hidden_sizes"]
    weights_dict = entry["weights"]

    hidden_Ws: List[np.ndarray] = []
    hidden_bs: List[np.ndarray] = []

    # Hidden layers
    for l_idx, _H in enumerate(hidden_sizes, start=1):
        layer_key = f"hidden_l{l_idx}"
        if layer_key not in weights_dict:
            raise KeyError(f"Layer key '{layer_key}' not found inside weights_dict")

        W_l = np.vstack(weights_dict[layer_key]["W"])
        b_l = np.vstack(weights_dict[layer_key]["b"]).reshape(-1)

        hidden_Ws.append(W_l)
        hidden_bs.append(b_l)

    # Output layer
    if "rul_output" not in weights_dict:
        raise KeyError("'rul_output' weights missing in weights_dict")

    W_out = np.vstack(weights_dict["rul_output"]["W"]).reshape(-1)
    b_out = np.vstack(weights_dict["rul_output"]["b"]).reshape(-1)

    hidden_Ws.append(W_out)
    hidden_bs.append(b_out)

    if verbose:
        print("Hidden sizes:", hidden_sizes)
        for i, (W, b) in enumerate(zip(hidden_Ws, hidden_bs), start=1):
            print(f"Layer {i}: W shape {W.shape}, b shape {b.shape}")

    return hidden_sizes, hidden_Ws, hidden_bs

# =========================
# Centroids (KMeans)
# =========================

def compute_trust_centroids(
    X_windows: np.ndarray,
    y: np.ndarray,
    n_centroids: int,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, KMeans]:
    """
    Compute KMeans centroids over flattened windows and return:
        centroids_mw: (C, M, W)
        y_c        : (C,)  mean target per cluster
        alpha      : (C,)  fraction of samples per cluster
        labels     : (N,)  cluster assignment per sample
        kmeans     : fitted KMeans object

    Parameters
    ----------
    X_windows : np.ndarray
        Array with shape (N, M, W).
    y : np.ndarray
        Target array with shape (N,).
    n_centroids : int
        Number of clusters (C).
    random_state : int
        Seed for KMeans reproducibility.
    """
    if X_windows.ndim != 3:
        raise ValueError(f"X_windows must have shape (N, M, W), got {X_windows.shape}")
    if X_windows.shape[0] != y.shape[0]:
        raise ValueError("X_windows and y must have the same number of samples (N).")

    N, M, W = X_windows.shape

    if n_centroids > N:
        raise ValueError(f"n_centroids={n_centroids} > N={N}")

    # Flatten windows: (N, M*W)
    X_flat = X_windows.reshape(N, M * W)

    kmeans = KMeans(
        n_clusters=n_centroids,
        random_state=random_state,
        n_init="auto",
    )
    kmeans.fit(X_flat)

    centroids_flat = kmeans.cluster_centers_  # (C, M*W)
    labels = kmeans.labels_                   # (N,)
    C = n_centroids

    # Back to (C, M, W)
    centroids_mw = centroids_flat.reshape(C, M, W)

    # Representative y per cluster + alpha proportions
    y_c = np.zeros(C, dtype=float)
    alpha_counts = np.zeros(C, dtype=int)

    for c in range(C):
        idx = np.where(labels == c)[0]
        alpha_counts[c] = len(idx)
        y_c[c] = float(y[idx].mean()) if alpha_counts[c] > 0 else float(np.mean(y))

    alpha = alpha_counts / float(N)

    return centroids_mw, y_c, alpha, labels, kmeans

# =========================
# MLP training config
# =========================

@dataclass(frozen=True)
class TrustMLPTrainConfig:
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 128
    verbose: int = 0
    seed: Optional[int] = RANDOM_SEED  # if you want to set a seed outside


# =========================
# Keras artifacts export
# =========================

def export_layer_weights_dict(model) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Export layer weights as:
        weights[layer_name] = {"W": W, "b": b}
    for layers that have weights.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) == 2:
            W, b = w
            out[layer.name] = {"W": W, "b": b}
    return out


def compute_final_metrics_regression(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    """
    Computes final regression metrics on train and validation sets.

    Metrics:
      - MSE
      - RMSE
      - MAE
      - R²
    """
    # Predictions
    y_train_pred = model.predict(X_train, verbose=0).reshape(-1)
    y_val_pred   = model.predict(X_val, verbose=0).reshape(-1)

    y_train_true = y_train.reshape(-1)
    y_val_true   = y_val.reshape(-1)

    # Train
    train_mse = mean_squared_error(y_train_true, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(y_train_true - y_train_pred))
    train_r2 = r2_score(y_train_true, y_train_pred)

    # Val
    val_mse = mean_squared_error(y_val_true, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = np.mean(np.abs(y_val_true - y_val_pred))
    val_r2 = r2_score(y_val_true, y_val_pred)

    return {
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "train_r2": float(train_r2),
        "val_r2": float(val_r2),
    }


# =========================
# MLP training (REBUILD)
# =========================

def train_and_collect(
    build_trust_mlp_fn,
    input_dim: int,
    hidden_sizes: Sequence[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Optional[TrustMLPTrainConfig] = None,
) -> Dict[str, Any]:
    """
    MLP_REBUILD:
    - Build a new TRUST-compatible MLP from scratch.
    - Train all layers.
    - Return metrics and weights for TRUST/MILP.

    Notes
    -----
    - This project version uses only train/val (no test set).
    - build_trust_mlp_fn is expected to return a compiled Keras model.
    """
    if cfg is None:
        cfg = TrustMLPTrainConfig()

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must be 2D arrays (N, input_dim).")
    if X_train.shape[1] != int(input_dim) or X_val.shape[1] != int(input_dim):
        raise ValueError("Input dimension mismatch between data and input_dim.")
    if y_train.shape[0] != X_train.shape[0] or y_val.shape[0] != X_val.shape[0]:
        raise ValueError("Target length mismatch with X arrays.")

    model = build_trust_mlp_fn(
        input_dim=int(input_dim),
        hidden_sizes=list(hidden_sizes),
        output_dim=1,
        learning_rate=float(cfg.learning_rate),
    )

    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        verbose=int(cfg.verbose),
    )

    history = history_obj.history
    final_metrics = compute_final_metrics_regression(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    weights = export_layer_weights_dict(model)

    return {
        "hidden_sizes": list(hidden_sizes),
        "model": model,
        "history": history,
        "weights": weights,                # used by extract_mlp_weights
        "weights_list": model.get_weights(),
        "final_metrics": final_metrics,
    }


# =========================
# MLP training (TRANSFER)
# =========================

def train_and_collect_transfer(
    build_trust_mlp_fn,
    input_dim: int,
    hidden_sizes: Sequence[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    prev_weights_list: Sequence[np.ndarray],
    prev_orig_indices: Sequence[int],
    curr_orig_indices: Sequence[int],
    cfg: Optional[TrustMLPTrainConfig] = None,
    init_new_feature_std: float = 0.05,
    first_dense_layer_index: int = 1,
) -> Dict[str, Any]:
    """
    MLP_TRANSFER:
    - Build a new TRUST-compatible MLP.
    - Initialize the first Dense layer using previous iteration weights aligned by orig_indices.
    - Copy remaining layer weights when shapes match.
    - Freeze all layers except the first Dense layer, then retrain only that layer.

    Notes
    -----
    - This project version uses only train/val (no test set).
    - first_dense_layer_index=1 assumes: [0]=InputLayer, [1]=first Dense (your builder matches this).
    """
    if cfg is None:
        cfg = TrustMLPTrainConfig()

    if prev_weights_list is None or prev_orig_indices is None or curr_orig_indices is None:
        raise ValueError("prev_weights_list, prev_orig_indices and curr_orig_indices must be provided for TRANSFER.")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must be 2D arrays (N, input_dim).")
    if X_train.shape[1] != int(input_dim) or X_val.shape[1] != int(input_dim):
        raise ValueError("Input dimension mismatch between data and input_dim.")
    if y_train.shape[0] != X_train.shape[0] or y_val.shape[0] != X_val.shape[0]:
        raise ValueError("Target length mismatch with X arrays.")

    model = build_trust_mlp_fn(
        input_dim=int(input_dim),
        hidden_sizes=list(hidden_sizes),
        output_dim=1,
        learning_rate=float(cfg.learning_rate),
    )

    # Retrieve new and old raw weights
    new_weights = model.get_weights()
    old_weights = list(prev_weights_list)

    if len(old_weights) < 2 or len(new_weights) < 2:
        raise ValueError("Unexpected Keras weight list format. Need at least [W0, b0, ...].")

    old_W0 = np.asarray(old_weights[0])
    old_b0 = np.asarray(old_weights[1]).reshape(-1)

    if old_W0.ndim != 2:
        raise ValueError(f"old_W0 must be 2D, got shape {old_W0.shape}")

    old_orig = np.asarray(prev_orig_indices, dtype=int).reshape(-1)
    new_orig = np.asarray(curr_orig_indices, dtype=int).reshape(-1)

    if new_orig.shape[0] != int(input_dim):
        raise ValueError("curr_orig_indices length must match input_dim.")

    # Build aligned first-layer weights
    hidden_1 = int(old_W0.shape[1])
    aligned_W0 = np.zeros((int(input_dim), hidden_1), dtype=old_W0.dtype)

    for j, orig_idx in enumerate(new_orig):
        matches = np.where(old_orig == int(orig_idx))[0]
        if matches.size > 0:
            old_pos = int(matches[0])
            aligned_W0[j, :] = old_W0[old_pos, :]
        else:
            aligned_W0[j, :] = np.random.normal(loc=0.0, scale=float(init_new_feature_std), size=(hidden_1,))

    # Replace first Dense weights/bias
    new_weights[0] = aligned_W0
    new_weights[1] = old_b0.copy()

    # Copy remaining weights when shapes match
    for k in range(2, len(new_weights)):
        if k < len(old_weights) and np.asarray(new_weights[k]).shape == np.asarray(old_weights[k]).shape:
            new_weights[k] = old_weights[k]

    model.set_weights(new_weights)

    # Freeze all layers except first Dense
    for i, layer in enumerate(model.layers):
        layer.trainable = (i == int(first_dense_layer_index))

    # Recompile because trainable flags changed
    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(cfg.learning_rate)),
        loss="mse",
        metrics=["mae"],
    )

    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        verbose=int(cfg.verbose),
    )

    history = history_obj.history
    final_metrics = compute_final_metrics_regression(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    weights = export_layer_weights_dict(model)

    return {
        "hidden_sizes": list(hidden_sizes),
        "model": model,
        "history": history,
        "weights": weights,                # used by extract_mlp_weights
        "weights_list": model.get_weights(),
        "final_metrics": final_metrics,
    }

# =========================
# Warm start helpers (MILP-1 -> MILP-2 and across iterations)
# =========================

def apply_warm_start_basic(curr: Dict[str, Any], last: Dict[str, Any]) -> None:
    """
    Warm start only for selection binaries:
    - Z0 always
    - S0 in sensors mode
    - W0 in windows mode

    Uses 'level' from last solution records and assigns it to `.value`.
    """
    mode = curr["mode"]

    # Z0 always exists
    last_Z0_df = last["Z0"].records.copy()

    if last_Z0_df is None or last_Z0_df.empty:
        return
    
    for _, row in last_Z0_df.iterrows():
        idx = int(row["m"])
        val = float(row["level"])
        curr["Z0"][idx].value = val

    # S0 only in sensors mode
    if mode == TRUST_MODE_SENSORS:
        last_S0_df = last["S0"].records.copy()
        for _, row in last_S0_df.iterrows():
            idx = int(row["s"])
            val = float(row["level"])
            curr["S0"][idx].value = val

    # W0 only in windows mode
    if mode == TRUST_MODE_WINDOWS:
        last_W0_df = last["W0"].records.copy()
        for _, row in last_W0_df.iterrows():
            idx = int(row["w"])
            val = float(row["level"])
            curr["W0"][idx].value = val


def apply_warm_start_sigmas(curr: Dict[str, Any], last: Dict[str, Any]) -> None:
    """
    Warm start sigma binaries (ReLU region indicators).

    In MILP-2, W1_var does not need explicit warm start because its initial level
    is already set during variable records creation.
    """
    sigma_curr_layers = curr["other_vars"]["sigma"]
    sigma_last_layers = last["other_vars"]["sigma"]

    for layer_idx, sigma_curr in enumerate(sigma_curr_layers):
        sigma_last = sigma_last_layers[layer_idx]
        last_df = sigma_last.records.copy()

        # Identify neuron set column (h1, h2, ...)
        h_col = [col for col in last_df.columns if str(col).startswith("h")][0]

        for _, row in last_df.iterrows():
            c_idx = int(row["c"])
            h_idx = int(row[h_col])
            val = float(row["level"])
            sigma_curr[c_idx, h_idx].value = val


def apply_warm_start(curr: Dict[str, Any], last: Dict[str, Any], use_sigmas: bool = True) -> None:
    """
    Complete warm start for TRUST:
    - Z0 always
    - S0 or W0 depending on mode
    - sigma binaries optionally
    """
    apply_warm_start_basic(curr, last)
    if use_sigmas:
        apply_warm_start_sigmas(curr, last)


# =========================
# Dataset flattening and feature mapping
# =========================

def flatten_windows_mw(X_mw: np.ndarray) -> np.ndarray:
    """
    Converts [N, M, W] to [N, M*W], keeping sensor-major order.
    """
    if X_mw.ndim != 3:
        raise ValueError(f"Expected X_mw to be 3D [N,M,W], got {X_mw.shape}")
    N, M, W = X_mw.shape
    return X_mw.reshape(N, M * W)


def flatten_datasets(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Flattens sliding-window datasets from (N, M, W) to (N, M*W).
    """
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (N,M,W), got shape {X_train.shape}")
    if X_val.ndim != 3:
        raise ValueError(f"X_val must be 3D (N,M,W), got shape {X_val.shape}")
    if X_test.ndim != 3:
        raise ValueError(f"X_test must be 3D (N,M,W), got shape {X_test.shape}")

    N_train, M, W = X_train.shape
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]

    X_train_flat = flatten_windows_mw(X_train)
    X_val_flat = flatten_windows_mw(X_val)
    X_test_flat = flatten_windows_mw(X_test)

    info = {
        "N_train": int(N_train),
        "N_val": int(N_val),
        "N_test": int(N_test),
        "M": int(M),
        "W": int(W),
        "input_dim": int(M * W),
    }

    return X_train_flat, X_val_flat, X_test_flat, info


def build_orig_indices(
    mode: str,
    M: int,
    W: int,
    M0: int,
    W0: int,
    active_features: Sequence[int],
    active_sensors: Sequence[int],
    active_windows: Sequence[int],
) -> np.ndarray:
    """
    Returns an array `orig_indices` of size M*W where each position j indicates
    which ORIGINAL flattened index (0-based) corresponds to the current local MLP
    feature column j.

    Flattening convention:
        local_flat_idx = s_local * W + w_local
    """
    if mode == TRUST_MODE_FULL:
        if len(active_features) != M * W:
            raise ValueError(f"FULL mode expects len(active_features)=M*W, got {len(active_features)} vs {M*W}")
        return np.asarray(active_features, dtype=int)

    if len(active_sensors) != M:
        raise ValueError(f"len(active_sensors) must match M={M}, got {len(active_sensors)}")
    if len(active_windows) != W:
        raise ValueError(f"len(active_windows) must match W={W}, got {len(active_windows)}")

    if mode not in (TRUST_MODE_SENSORS, TRUST_MODE_WINDOWS):
        raise ValueError(f"Invalid mode={mode}")

    orig_indices = np.empty(M * W, dtype=int)
    k = 0
    for s_orig in active_sensors:
        for w_orig in active_windows:
            orig_indices[k] = int(s_orig) * int(W0) + int(w_orig)
            k += 1

    return orig_indices


# =========================
# Importance computation
# =========================

from pandas.api.types import is_numeric_dtype, is_categorical_dtype

def _compute_imp_features_orig_from_W1(
    model_pack_2: Dict[str, Any],
    local_to_orig: np.ndarray,
    n_features_original: int,
) -> np.ndarray:
    """
    Compute importance per ORIGINAL flat feature index from MILP-2 first-layer weights.

    Importance of a LOCAL feature f:
        I_local[f] = sum_h |W1[f, h]|
    """
    W1_var = model_pack_2["other_vars"]["W1_var"]
    W1_df = W1_var.records.copy()

    if "m" not in W1_df.columns:
        raise KeyError(f"W1_var.records missing column 'm'. Columns={W1_df.columns.tolist()}")

    s = W1_df["m"]

    # Robust coercion for GAMSPy UELs (often categorical strings like '1','2',...)
    if is_categorical_dtype(s):
        s = s.astype(str)

    if not is_numeric_dtype(s):
        # try numeric conversion (strings -> int)
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.isna().any():
            bad = W1_df.loc[s_num.isna(), "m"].head(10).tolist()
            raise ValueError(
                f"Column 'm' could not be converted to numeric. Sample bad values={bad}"
            )
        s = s_num

    W1_df["m_idx"] = s.astype(int)
    W1_df["m_local0"] = W1_df["m_idx"] - 1

    # Now compute local importance from first-layer variable levels
    imp_local = (
        W1_df.groupby("m_local0")["level"]
        .apply(lambda x: float(np.sum(np.abs(np.asarray(x, dtype=float)))))
    )

    imp_orig = np.zeros(int(n_features_original), dtype=float)
    for m_local0, val in imp_local.items():
        i_orig = int(local_to_orig[int(m_local0)])
        imp_orig[i_orig] = float(val)

    return imp_orig


def compute_importance_full_milp2(model_pack_2: Dict[str, Any], M0: int, W0: int, active_features: Sequence[int]):
    """
    FULL mode: importance per original flat feature (M0*W0).
    """
    n_features_original = int(M0) * int(W0)
    local_to_orig = np.asarray(active_features, dtype=int)

    imp_orig = _compute_imp_features_orig_from_W1(model_pack_2, local_to_orig, n_features_original)

    imp_norm = np.zeros_like(imp_orig, dtype=float)
    mask = imp_orig > 0
    if np.any(mask):
        imp_norm[mask] = imp_orig[mask] / imp_orig[mask].sum()

    return imp_orig, imp_norm


def compute_importance_sensors_milp2(model_pack_2: Dict[str, Any], M0: int, W0: int, active_sensors: Sequence[int]):
    """
    SENSORS mode: importance per original sensor (M0).
    """
    active_sensors = np.asarray(active_sensors, dtype=int)
    M_curr = len(active_sensors)
    W_curr = int(W0)

    n_local_features = M_curr * W_curr
    local_to_orig = np.zeros(n_local_features, dtype=int)

    idx = 0
    for s_orig in active_sensors:
        for w_orig in range(W0):
            local_to_orig[idx] = int(s_orig) * int(W0) + int(w_orig)
            idx += 1

    n_features_original = int(M0) * int(W0)
    imp_features_orig = _compute_imp_features_orig_from_W1(model_pack_2, local_to_orig, n_features_original)

    sensor_imp = np.zeros(int(M0), dtype=float)
    for f_orig, val in enumerate(imp_features_orig):
        if val == 0.0:
            continue
        s_idx = int(f_orig) // int(W0)
        sensor_imp[s_idx] += float(val)

    sensor_norm = np.zeros_like(sensor_imp, dtype=float)
    mask = sensor_imp > 0
    if np.any(mask):
        sensor_norm[mask] = sensor_imp[mask] / sensor_imp[mask].sum()

    return sensor_imp, sensor_norm


def compute_importance_windows_milp2(model_pack_2: Dict[str, Any], M0: int, W0: int, active_windows: Sequence[int]):
    """
    WINDOWS mode: importance per original window (W0).
    """
    active_windows = np.asarray(active_windows, dtype=int)
    M_curr = int(M0)
    W_curr = len(active_windows)

    n_local_features = M_curr * W_curr
    local_to_orig = np.zeros(n_local_features, dtype=int)

    idx = 0
    for s_orig in range(M0):
        for w_orig in active_windows:
            local_to_orig[idx] = int(s_orig) * int(W0) + int(w_orig)
            idx += 1

    n_features_original = int(M0) * int(W0)
    imp_features_orig = _compute_imp_features_orig_from_W1(model_pack_2, local_to_orig, n_features_original)

    window_imp = np.zeros(int(W0), dtype=float)
    for f_orig, val in enumerate(imp_features_orig):
        if val == 0.0:
            continue
        w_idx = int(f_orig) % int(W0)
        window_imp[w_idx] += float(val)

    window_norm = np.zeros_like(window_imp, dtype=float)
    mask = window_imp > 0
    if np.any(mask):
        window_norm[mask] = window_imp[mask] / window_imp[mask].sum()

    return window_imp, window_norm


def compute_importance_full_from_mlp(hidden_Ws: List[np.ndarray], M0: int, W0: int, active_features: Sequence[int]):
    """
    Importance per original flat feature computed directly from trained MLP first-layer weights.

    This keeps comparability across architectures by using only layer 1.
    """
    active_features = np.asarray(active_features, dtype=int)
    n_features_original = int(M0) * int(W0)

    W1 = np.asarray(hidden_Ws[0], dtype=float)
    imp_local = np.sum(np.abs(W1), axis=1)

    imp_orig = np.zeros(n_features_original, dtype=float)
    for i_local, val in enumerate(imp_local):
        i_orig = int(active_features[i_local])
        imp_orig[i_orig] = float(val)

    imp_norm = np.zeros_like(imp_orig, dtype=float)
    mask = imp_orig > 0
    if np.any(mask):
        imp_norm[mask] = imp_orig[mask] / imp_orig[mask].sum()

    return imp_orig, imp_norm

def compute_importance_sensors_from_mlp(
    hidden_Ws: List[np.ndarray],
    M0: int,
    W0: int,
    active_sensors: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SENSORS mode (baseline / surrogate MLP):
    - Computes importance per ORIGINAL sensor index (size M0).
    - Uses ONLY the first-layer weights of the MLP to keep comparability.

    Steps:
      1) Local feature importance: imp_local_feature[i] = sum_h |W1[i, h]|
         where i runs over local flat features (M_curr * W0).
      2) Map each local flat feature to original flat feature:
         orig_flat = s_orig*W0 + w_orig
      3) Aggregate per original sensor: sum over windows
      4) Normalize over sensors with non-zero importance.
    """
    active_sensors = np.asarray(active_sensors, dtype=int)
    M_curr = int(active_sensors.shape[0])
    W0 = int(W0)

    W1 = np.asarray(hidden_Ws[0], dtype=float)  # (input_dim_local, hidden_1)
    imp_local_feature = np.sum(np.abs(W1), axis=1)  # (M_curr*W0,)

    expected_dim = M_curr * W0
    if imp_local_feature.shape[0] != expected_dim:
        raise ValueError(
            f"SENSORS from MLP: expected input_dim={expected_dim} (M_curr*W0) "
            f"but got {imp_local_feature.shape[0]}. "
            f"M_curr={M_curr}, W0={W0}."
        )

    sensor_imp = np.zeros(int(M0), dtype=float)

    idx = 0
    for s_orig in active_sensors:
        s_orig_i = int(s_orig)
        for _w_orig in range(W0):
            sensor_imp[s_orig_i] += float(imp_local_feature[idx])
            idx += 1

    sensor_norm = np.zeros_like(sensor_imp, dtype=float)
    mask = sensor_imp > 0
    if np.any(mask):
        sensor_norm[mask] = sensor_imp[mask] / float(sensor_imp[mask].sum())

    return sensor_imp, sensor_norm


def compute_importance_windows_from_mlp(
    hidden_Ws: List[np.ndarray],
    M0: int,
    W0: int,
    active_windows: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    WINDOWS mode (baseline / surrogate MLP):
    - Computes importance per ORIGINAL window index (size W0).
    - Uses ONLY the first-layer weights of the MLP.

    Steps:
      1) Local feature importance: imp_local_feature[i] = sum_h |W1[i, h]|
         where i runs over local flat features (M0 * W_curr).
      2) Map each local flat feature to original flat feature:
         orig_flat = s_orig*W0 + w_orig
      3) Aggregate per original window: sum over sensors
      4) Normalize over windows with non-zero importance.
    """
    active_windows = np.asarray(active_windows, dtype=int)
    W_curr = int(active_windows.shape[0])
    M0 = int(M0)
    W0 = int(W0)

    W1 = np.asarray(hidden_Ws[0], dtype=float)
    imp_local_feature = np.sum(np.abs(W1), axis=1)  # (M0*W_curr,)

    expected_dim = M0 * W_curr
    if imp_local_feature.shape[0] != expected_dim:
        raise ValueError(
            f"WINDOWS from MLP: expected input_dim={expected_dim} (M0*W_curr) "
            f"but got {imp_local_feature.shape[0]}. "
            f"M0={M0}, W_curr={W_curr}."
        )

    window_imp = np.zeros(int(W0), dtype=float)

    idx = 0
    for _s_orig in range(M0):
        for w_orig in active_windows:
            window_imp[int(w_orig)] += float(imp_local_feature[idx])
            idx += 1

    window_norm = np.zeros_like(window_imp, dtype=float)
    mask = window_imp > 0
    if np.any(mask):
        window_norm[mask] = window_imp[mask] / float(window_imp[mask].sum())

    return window_imp, window_norm

# =========================
# Pretty printing (optional)
# =========================

def pretty_mlp_progress(n0_value: int, final_metrics: dict) -> None:
    def _fmt(x):
        return "NA" if x is None else f"{float(x):.6f}"

    tr_mse = final_metrics.get("train_mse")
    va_mse = final_metrics.get("val_mse")
    tr_rmse = final_metrics.get("train_rmse")
    va_rmse = final_metrics.get("val_rmse")
    tr_mae = final_metrics.get("train_mae")
    va_mae = final_metrics.get("val_mae")
    tr_r2 = final_metrics.get("train_r2")
    va_r2 = final_metrics.get("val_r2")

    print("=" * 70)
    print(f" ITERATION  N0 = {n0_value}")
    print("-" * 70)
    print(" MLP metrics (regression):")
    print(f"   Train: MSE={_fmt(tr_mse)}, RMSE={_fmt(tr_rmse)}, MAE={_fmt(tr_mae)}, R2={_fmt(tr_r2)}")
    print(f"   Val  : MSE={_fmt(va_mse)}, RMSE={_fmt(va_rmse)}, MAE={_fmt(va_mae)}, R2={_fmt(va_r2)}")


def pretty_milp_df(title: str, df: "pd.DataFrame") -> None:
    """
    Pretty print for the MILP summary DataFrame (1-row).
    """
    row = df.iloc[0]

    print("\n" + "=" * 70)
    print(f" {title}")
    print("-" * 70)
    print(f" Solver Status:     {row.get('Solver Status', '-')}")
    print(f" Model Status:      {row.get('Model Status', '-')}")
    print(f" Objective:         {row.get('Objective', '-')}")
    print(f" Equations:         {row.get('Num of Equations', '-')}")
    print(f" Variables:         {row.get('Num of Variables', '-')}")
    print(f" Model Type:        {row.get('Model Type', '-')}")
    print(f" Solver:            {row.get('Solver', '-')}")
    print(f" Solver Time (s):   {row.get('Solver Time', '-')}")
    print("=" * 70 + "\n")


def pretty_selection_progress(
    n0_value: int,
    selection_vector: np.ndarray,
    selected_sensors: Optional[Sequence[int]] = None,
    selected_windows: Optional[Sequence[int]] = None,
) -> None:
    """
    Pretty print for the selection vector and optional sensor/window info.
    """
    print("\n" + "-" * 70)
    print(f" SELECTION SUMMARY  (N0 = {n0_value})")
    print("-" * 70)

    cols = 30
    vec = selection_vector.astype(int).tolist()
    for i in range(0, len(vec), cols):
        chunk = vec[i:i + cols]
        idx = list(range(i, min(len(vec), i + cols)))
        print(f"  {idx}: {chunk}")

    active_count = int(np.sum(selection_vector))
    total_count = int(len(selection_vector))
    print(f"\n Active count = {active_count} / {total_count}")

    if selected_sensors is not None:
        print(f" Selected sensors (1-based): {np.array(selected_sensors)}")
    if selected_windows is not None:
        print(f" Selected windows (1-based): {np.array(selected_windows)}")

    print("-" * 70 + "\n")

def mlp_forward_numpy(
    X: np.ndarray,
    hidden_Ws: List[np.ndarray],
    hidden_bs: List[np.ndarray],
) -> np.ndarray:
    """
    Forward pass for a TRUST-compatible MLP using numpy.

    Assumptions:
    - hidden_Ws = [W1, W2, ..., W_out]
      where W_out has shape (H_last,) for scalar output
    - hidden_bs = [b1, b2, ..., b_out]
      where b_out is scalar-like (shape (1,) or ())
    - ReLU on hidden layers, linear output.

    Parameters
    ----------
    X : np.ndarray
        Input matrix (N, input_dim).
    hidden_Ws : list[np.ndarray]
        Weight matrices for hidden layers and output vector for last layer.
    hidden_bs : list[np.ndarray]
        Bias vectors for hidden layers and scalar bias for output.

    Returns
    -------
    y_pred : np.ndarray
        Predictions with shape (N,).
    """
    a = np.asarray(X, dtype=float)

    n_layers = len(hidden_Ws)
    if len(hidden_bs) != n_layers:
        raise ValueError("hidden_Ws and hidden_bs must have same length.")

    # Hidden layers (all except last)
    for l in range(n_layers - 1):
        W = np.asarray(hidden_Ws[l], dtype=float)
        b = np.asarray(hidden_bs[l], dtype=float).reshape(-1)
        a = a @ W + b
        a = np.maximum(0.0, a)  # ReLU

    # Output layer (linear)
    W_out = np.asarray(hidden_Ws[-1], dtype=float).reshape(-1)     # (H_last,)
    b_out = float(np.asarray(hidden_bs[-1], dtype=float).reshape(-1)[0])
    y_pred = a @ W_out + b_out

    return y_pred.reshape(-1)


def centroid_weighted_mae(
    centroids_mw: np.ndarray,
    y_c: np.ndarray,
    alpha: np.ndarray,
    hidden_Ws: List[np.ndarray],
    hidden_bs: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute the weighted MAE surrogate over centroids:
        sum_c alpha[c] * |y_pred[c] - y_c[c]|

    Parameters
    ----------
    centroids_mw : np.ndarray
        Centroids with shape (C, M, W).
    y_c : np.ndarray
        Representative target per centroid, shape (C,).
    alpha : np.ndarray
        Cluster weights (fractions), shape (C,). Ideally sum(alpha)=1.
    hidden_Ws, hidden_bs
        MLP weights in TRUST format.

    Returns
    -------
    dict with mae, sum_alpha, and optional diagnostic stats.
    """
    C, M, W = centroids_mw.shape
    centroids_flat = centroids_mw.reshape(C, M * W)

    y_pred = mlp_forward_numpy(centroids_flat, hidden_Ws, hidden_bs)

    y_c = np.asarray(y_c, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    if y_c.shape[0] != C or alpha.shape[0] != C:
        raise ValueError("y_c and alpha must match number of centroids C.")

    abs_err = np.abs(y_pred - y_c)
    w_mae = float(np.sum(alpha * abs_err))
    sum_alpha = float(np.sum(alpha))

    return {
        "centroid_weighted_mae": w_mae,
        "sum_alpha": sum_alpha,
        "mean_abs_err": float(abs_err.mean()),
        "max_abs_err": float(abs_err.max()),
    }


# =========================
# I/O suppression (optional)
# =========================

@contextlib.contextmanager
def suppress_gams_warnings():
    """
    Suppress stdout and stderr during solver calls.
    """
    with open(os.devnull, "w") as devnull:
        oldout, olderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = oldout, olderr

