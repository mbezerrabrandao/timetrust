# trust/pipeline.py
from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from gamspy import Options

from public_time_trust.utils.milp_utils import (
    TRUST_MODE_FULL,
    TRUST_MODE_SENSORS,
    TRUST_MODE_WINDOWS,
    MILP_MODE_1,
    MILP_MODE_2,
    MLP_REBUILD,
    MLP_TRANSFER,
    BASELINE_MODE_NONE,
    BASELINE_MODE_CAPPED,
)

from public_time_trust.utils.trust_utils import (
    apply_warm_start,
    build_orig_indices,
    centroid_weighted_mae,
    compute_importance_full_from_mlp,
    compute_importance_full_milp2,
    compute_importance_sensors_from_mlp,
    compute_importance_sensors_milp2,
    compute_importance_windows_from_mlp,
    compute_importance_windows_milp2,
    compute_trust_centroids,
    extract_mlp_weights,
    flatten_datasets,
    pretty_milp_df,
    pretty_mlp_progress,
    pretty_selection_progress,
    suppress_gams_warnings,
    train_and_collect,
    train_and_collect_transfer,
    TrustMLPTrainConfig,
)

from public_time_trust.utils.mlp_utils import build_trust_mlp, RANDOM_SEED
from public_time_trust.milp.trust_milp import full_variable_layer_trust

from public_time_trust.pipeline_artifacts import finalize_trust_run_artifacts

# =========================
# Config and persistence
# =========================

@dataclass(frozen=True)
class TrustRunConfig:
    dataset_name: str
    window_tag: str
    hidden_layers: Tuple[int, ...]
    mode: str
    mlp_mode: str

    C: int
    beta: float
    milp_time_cap: int

    baseline_mode: str
    baseline_slack: float

    learning_rate: float
    epochs: int
    batch_size: int
    seed: int = 2026


def _arch_tag(hidden_layers: Sequence[int]) -> str:
    return "h" + "_".join(str(x) for x in hidden_layers)


def _run_tag(cfg: TrustRunConfig) -> str:
    return f"{cfg.mode}__{cfg.mlp_mode}__{cfg.baseline_mode}_slack{cfg.baseline_slack:.3f}"


def _default_results_dir(cfg: TrustRunConfig, root: Optional[Path] = None) -> Path:
    if root is None:
        root = Path.cwd()
    return (
        root
        / "results"
        / cfg.dataset_name
        / cfg.window_tag
        / _arch_tag(cfg.hidden_layers)
        / _run_tag(cfg)
    )


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def _save_run_metadata(out_dir: Path, cfg: TrustRunConfig, extra: Optional[Dict[str, Any]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_config": _json_safe(asdict(cfg))}
    if extra is not None:
        payload["run_extra"] = _json_safe(extra)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_iteration_checkpoint(out_dir: Path, iter_idx: int, iter_info: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    for k, dtype in (
        ("selection_vector", np.int8),
        ("importance_vector", np.float32),
        ("importance_vector_norm", np.float32),
        ("selected_original", np.int32),
        ("orig_indices", np.int32),
    ):
        if k in iter_info and iter_info[k] is not None:
            arrays[k] = np.asarray(iter_info[k], dtype=dtype)

    np.savez_compressed(out_dir / f"iter_{iter_idx:03d}.npz", **arrays)

    meta = dict(iter_info)
    for k in ("summary_1", "summary_2"):
        if isinstance(meta.get(k, None), pd.DataFrame):
            meta[k] = meta[k].to_dict(orient="records")
    if isinstance(meta.get("selection_df", None), pd.DataFrame):
        meta["selection_df"] = meta["selection_df"].to_dict(orient="records")

    meta.pop("model_pack_1", None)
    meta.pop("model_pack_2", None)

    with open(out_dir / f"iter_{iter_idx:03d}.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2)

def _save_mlp_weights_checkpoint(out_dir: Path, iter_idx: int, weights_list: Sequence[np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {f"w_{i:03d}": np.asarray(w) for i, w in enumerate(weights_list)}
    np.savez_compressed(out_dir / f"iter_{iter_idx:03d}_mlp_weights.npz", **payload)

def _find_last_iteration(out_dir: Path) -> Optional[int]:
    if not out_dir.exists():
        return None
    iters = sorted(out_dir.glob("iter_*.json"))
    if not iters:
        return None

    # iter_001.json -> 1
    last = iters[-1].stem  # "iter_001"
    try:
        return int(last.split("_")[1])
    except Exception:
        return None


def _load_iteration_meta(out_dir: Path, iter_idx: int) -> Dict[str, Any]:
    with open(out_dir / f"iter_{iter_idx:03d}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def _list_iteration_json(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("iter_*.json"))

def _load_all_iterations(out_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in _list_iteration_json(out_dir):
        with open(p, "r", encoding="utf-8") as f:
            items.append(json.load(f))
    # Orden por iter_idx si existe, si no por filename
    items.sort(key=lambda d: int(d.get("iter_idx", 10**9)))
    return items

def _find_last_selection_iteration(out_dir: Path) -> Optional[int]:
    """
    Returns the last iteration index that contains a selection (selected_original).
    Skips final MLP checkpoint (n0_value=0) and any incomplete checkpoints.
    """
    last = _find_last_iteration(out_dir)
    if last is None:
        return None

    for i in range(int(last), 0, -1):
        try:
            meta = _load_iteration_meta(out_dir, i)
        except Exception:
            continue
        if isinstance(meta, dict) and "selected_original" in meta:
            return i
    return None

def _load_mlp_weights_if_present(out_dir: Path, iter_idx: int) -> Optional[List[np.ndarray]]:
    fp = out_dir / f"iter_{iter_idx:03d}_mlp_weights.npz"
    if not fp.exists():
        return None

    data = np.load(fp, allow_pickle=False)
    # restore in order w_000, w_001, ...
    keys = sorted(data.files)
    return [np.asarray(data[k]) for k in keys]


def _restore_active_indices_from_checkpoint(
    *,
    mode: str,
    M0: int,
    W0: int,
    selected_original: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild active_sensors / active_windows / active_features in ORIGINAL index space
    from the selection saved in the last completed iteration.
    """
    selected_original = np.asarray(selected_original, dtype=int).reshape(-1)

    if mode == TRUST_MODE_FULL:
        active_features = selected_original.copy()
        active_sensors = np.unique(active_features // W0).astype(int)
        active_windows = np.unique(active_features % W0).astype(int)
        return active_sensors, active_windows, active_features

    if mode == TRUST_MODE_SENSORS:
        active_sensors = selected_original.copy()
        active_windows = np.arange(W0, dtype=int)
        active_features = np.arange(M0 * W0, dtype=int)  # unused in sensors mode, but keep defined
        return active_sensors, active_windows, active_features

    if mode == TRUST_MODE_WINDOWS:
        active_windows = selected_original.copy()
        active_sensors = np.arange(M0, dtype=int)
        active_features = np.arange(M0 * W0, dtype=int)  # unused in windows mode, but keep defined
        return active_sensors, active_windows, active_features

    raise ValueError(f"Invalid mode={mode}")


def _apply_selection_to_datasets(
    *,
    mode: str,
    X_train0: np.ndarray,
    X_val0: np.ndarray,
    M0: int,
    W0: int,
    active_sensors: np.ndarray,
    active_windows: np.ndarray,
    active_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebuild the CURRENT (filtered) X_train/X_val directly from the ORIGINAL datasets
    and the current active indices.
    """
    if mode == TRUST_MODE_SENSORS:
        X_train = X_train0[:, active_sensors, :]
        X_val = X_val0[:, active_sensors, :]
        return X_train, X_val

    if mode == TRUST_MODE_WINDOWS:
        X_train = X_train0[:, :, active_windows]
        X_val = X_val0[:, :, active_windows]
        return X_train, X_val

    if mode == TRUST_MODE_FULL:
        # active_features are original flat indices in [0, M0*W0)
        s_idx = (active_features // W0).astype(int)
        w_idx = (active_features % W0).astype(int)

        # build (N, n_feat) then reshape to (N, n_feat, 1) as your FULL convention
        X_train_flat = X_train0[:, s_idx, w_idx]
        X_val_flat = X_val0[:, s_idx, w_idx]

        X_train = X_train_flat.reshape(X_train0.shape[0], -1, 1)
        X_val = X_val_flat.reshape(X_val0.shape[0], -1, 1)
        return X_train, X_val

    raise ValueError(f"Invalid mode={mode}")


# =========================
# Baseline cap helpers
# =========================

def _baseline_floor_from_slack(baseline_mae: float, slack: float) -> float:
    """
    slack = fraction of allowed improvement relative to baseline.
      slack=0.10 means: MILP surrogate can be up to 10% better than baseline.
    floor = baseline_mae * (1 - slack)
    """
    if baseline_mae <= 0:
        raise ValueError(f"baseline_mae must be > 0. Got {baseline_mae}")
    if slack < 0 or slack >= 1.0:
        raise ValueError("baseline_slack must be in [0, 1).")
    return float(baseline_mae) * (1.0 - float(slack))


def _project_centroids_full_to_local(
    centroids_mw_full: np.ndarray,
    mode: str,
    active_features: np.ndarray,
    active_sensors: np.ndarray,
    active_windows: np.ndarray,
) -> np.ndarray:
    """
    Project fixed centroids (C, M0, W0) into current local space depending on TRUST mode.

    Returns
    -------
    centroids_mw_local : np.ndarray
        - FULL    -> (C, M, W) where M*W = len(active_features) and W=1 (packed later) OR
                    (C, M, W) matching current X_train shape after filtering.
        - SENSORS -> (C, M, W0) where M=len(active_sensors), W=W0
        - WINDOWS -> (C, M0, W) where W=len(active_windows), M=M0
    """
    C, M0, W0 = centroids_mw_full.shape

    if mode == TRUST_MODE_FULL:
        # active_features are original flat indices -> convert to (s,w)
        s_idx = (active_features // W0).astype(int)
        w_idx = (active_features % W0).astype(int)
        local = centroids_mw_full[:, s_idx, w_idx]  # (C, n_feat)
        # represent as (C, n_feat, 1) to match your FULL flattening contract downstream
        return local.reshape(C, local.shape[1], 1)

    if mode == TRUST_MODE_SENSORS:
        # keep all windows, only subset sensors
        return centroids_mw_full[:, active_sensors.astype(int), :]

    if mode == TRUST_MODE_WINDOWS:
        # keep all sensors, only subset windows
        return centroids_mw_full[:, :, active_windows.astype(int)]

    raise ValueError(f"Invalid mode={mode}")


def _compute_baseline_centroid_mae_fixed(
    centroids_mw_full: np.ndarray,
    y_c: np.ndarray,
    alpha: np.ndarray,
    baseline_hidden_Ws: List[np.ndarray],
    baseline_hidden_bs: List[np.ndarray],
) -> float:
    """
    Baseline MAE surrogate computed once on fixed centroids in original space.
    """
    stats = centroid_weighted_mae(
        centroids_mw=centroids_mw_full,
        y_c=y_c,
        alpha=alpha,
        hidden_Ws=baseline_hidden_Ws,
        hidden_bs=baseline_hidden_bs,
    )
    return float(stats["centroid_weighted_mae"])

def _flatten_train_val(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if X_train.ndim != 3 or X_val.ndim != 3:
        raise ValueError("Expected X_train and X_val as 3D arrays (N,M,W).")
    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Train/val feature shapes must match.")
    N_train, M, W = X_train.shape
    X_train_flat = X_train.reshape(N_train, M * W)
    X_val_flat = X_val.reshape(X_val.shape[0], M * W)
    return X_train_flat, X_val_flat, {"M": int(M), "W": int(W), "input_dim": int(M * W)}


# =========================
# Main TRUST pipeline
# =========================

def execute_trust_algorithm(
    *,
    dataset_name: str,
    window_tag: str,
    results_root: Optional[Path] = None,

    # TRUST parameters
    hidden_layers: Sequence[int] = (10,),
    mode: str = TRUST_MODE_SENSORS,
    C: int = 50,
    beta: float = 0.5,

    # Surrogate MLP mode (NOT baseline)
    mlp_mode: str = MLP_REBUILD,
    milp_time_cap: int = 60,

    learning_rate: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,

    # Baseline cap
    baseline_mode: str = BASELINE_MODE_NONE,
    baseline_slack: float = 0.0,

    # Data (preprocessed) - train/val only
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,

    # Baseline MLP (fixed)
    baseline_hidden_Ws: Optional[List[np.ndarray]] = None,
    baseline_hidden_bs: Optional[List[np.ndarray]] = None,

    baseline_metrics: Optional[Dict[str, float]] = None,
    baseline_weights_list: Optional[List[np.ndarray]] = None,

    # Random seed
    seed: int = RANDOM_SEED,
    verbose: bool = True,

    # Checkpointing
    resume: bool = True,
) -> Dict[str, Any]:
    """
    TRUST pipeline (fixed-centroids + fixed baseline cap):
    - Compute centroids once in the original space (C, M0, W0) using X_train.
    - Compute baseline centroid MAE once (if capped).
    - Iterate n0 from max to 1:
        * Project fixed centroids into current local space (based on active indices).
        * Train surrogate MLP on current local dataset (REBUILD or TRANSFER).
        * Solve MILP-1 then MILP-2 (MILP-2 optionally capped).
        * Extract selection, update active indices, filter datasets, save checkpoint.
    """
    cfg = TrustRunConfig(
        dataset_name=dataset_name,
        window_tag=window_tag,
        hidden_layers=tuple(int(x) for x in hidden_layers),
        mode=mode,
        mlp_mode=mlp_mode,
        C=int(C),
        beta=float(beta),
        milp_time_cap=int(milp_time_cap),
        baseline_mode=baseline_mode,
        baseline_slack=float(baseline_slack),
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        batch_size=int(batch_size),
        seed=int(seed),
    )

    out_dir = _default_results_dir(cfg, root=results_root)

    if baseline_mode == BASELINE_MODE_CAPPED:
        if baseline_hidden_Ws is None or baseline_hidden_bs is None:
            raise ValueError("CAPPED mode requires baseline_hidden_Ws and baseline_hidden_bs.")

    # Defensive copies
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Original dimensions (fixed-length vectors)
    M0 = int(X_train.shape[1])
    W0 = int(X_train.shape[2])
    
    # -----------------------------
    # 0) Fixed centroids + fixed cap
    # -----------------------------
    centroids_full, y_c, alpha, labels, kmeans = compute_trust_centroids(
        X_windows=X_train,
        y=y_train,
        n_centroids=int(C),
        random_state=int(seed),
    )

    baseline_mae_clusters = None
    baseline_mae_floor = None

    if baseline_mode == BASELINE_MODE_CAPPED:
        baseline_mae_clusters = _compute_baseline_centroid_mae_fixed(
            centroids_mw_full=centroids_full,
            y_c=y_c,
            alpha=alpha,
            baseline_hidden_Ws=baseline_hidden_Ws,
            baseline_hidden_bs=baseline_hidden_bs,
        )
        baseline_mae_floor = _baseline_floor_from_slack(
            baseline_mae_clusters,
            baseline_slack,
        )

        print(
            "[BASELINE | CAPPED]\n"
            f"  centroid MAE        : {baseline_mae_clusters:.6f}\n"
            f"  slack               : {baseline_slack:.3f}\n"
            f"  MAE floor (enforced): {baseline_mae_floor:.6f}"
        )

    
    _save_run_metadata(
        out_dir,
        cfg,
        extra={
            "M0": M0,
            "W0": W0,
            "baseline_mae_clusters": baseline_mae_clusters,
            "baseline_mae_floor": baseline_mae_floor,
        },
    )

    # ------------------------------------------------------------------
    # Keep originals to allow exact reconstruction on resume
    # ------------------------------------------------------------------
    X_train0 = X_train.copy()
    X_val0 = X_val.copy()

    # Original dimensions (fixed for the whole run)
    M0 = int(X_train0.shape[1])
    W0 = int(X_train0.shape[2])

    # ------------------------------------------------------------------
    # Loop schedule (defined ONCE from the original dataset)
    # ------------------------------------------------------------------
    if mode == TRUST_MODE_FULL:
        max_n0 = M0 * W0
    elif mode == TRUST_MODE_SENSORS:
        max_n0 = M0
    elif mode == TRUST_MODE_WINDOWS:
        max_n0 = W0
    else:
        raise ValueError(f"Invalid mode={mode}")

    # ------------------------------------------------------------------
    # Default (fresh) state in ORIGINAL index space
    # ------------------------------------------------------------------
    active_sensors = np.arange(M0, dtype=int)
    active_windows = np.arange(W0, dtype=int)
    active_features = np.arange(M0 * W0, dtype=int)

    # TRANSFER state (surrogate only)
    prev_model_weights_list: Optional[List[np.ndarray]] = None
    prev_orig_indices: Optional[np.ndarray] = None

    # Resume cursor
    start_iter_idx = 1
    start_n0 = max_n0 - 1

    # ------------------------------------------------------------------
    # Resume logic (if results already exist)
    # ------------------------------------------------------------------
    if resume:
        last_iter = _find_last_selection_iteration(out_dir)
        if last_iter is not None and int(last_iter) >= 1:
            last_iter = int(last_iter)
            last_meta = _load_iteration_meta(out_dir, last_iter)

            # Restore active indices from last selection (original index space)
            selected_original = np.asarray(last_meta["selected_original"], dtype=int)

            active_sensors, active_windows, active_features = _restore_active_indices_from_checkpoint(
                mode=mode,
                M0=M0,
                W0=W0,
                selected_original=selected_original,
            )

            # Rebuild current datasets exactly from originals
            X_train, X_val = _apply_selection_to_datasets(
                mode=mode,
                X_train0=X_train0,
                X_val0=X_val0,
                M0=M0,
                W0=W0,
                active_sensors=active_sensors,
                active_windows=active_windows,
                active_features=active_features,
            )

            # Restore TRANSFER state if weights are available
            prev_model_weights_list = _load_mlp_weights_if_present(out_dir, last_iter)
            prev_orig_indices = np.asarray(last_meta["orig_indices"], dtype=int)

            # Continue from next n0 value
            start_iter_idx = last_iter + 1
            start_n0 = int(last_meta["n0_value"]) - 1

            if verbose:
                print(f"[RESUME] Found checkpoint iter={last_iter:03d}. Continuing at N0={start_n0}.")
                if mlp_mode == MLP_TRANSFER and prev_model_weights_list is None:
                    print("[RESUME] No saved surrogate weights found. TRANSFER will behave like REBUILD on the next step.")

    # If resume landed us below 1, nothing to do
    if start_n0 < 1:
        if verbose:
            print("[RESUME] Nothing left to run (start_n0 < 1).")
        return {
            "out_dir": str(out_dir),
            "run_config": asdict(cfg),
            "history": [],
        }

    # --------------------------------------------------
    # Baseline iter_000 (selection + importance must match MODE)
    # --------------------------------------------------
    active_sensors0 = np.arange(M0, dtype=int)
    active_windows0 = np.arange(W0, dtype=int)
    active_features0 = np.arange(M0 * W0, dtype=int)

    # orig_indices for baseline context (local space == original space at iter_000,
    # but in SENSORS/WINDOWS we still need the local_flat -> orig_flat mapping)
    if mode == TRUST_MODE_FULL:
        orig_indices0 = active_features0.copy()
        input_dim0 = int(M0 * W0)

        baseline_selection = np.ones(M0 * W0, dtype=int)
        selected_original = active_features0.copy()

        baseline_importance, _ = compute_importance_full_from_mlp(
            hidden_Ws=baseline_hidden_Ws,
            M0=M0,
            W0=W0,
            active_features=active_features0,
        )

    elif mode == TRUST_MODE_SENSORS:
        orig_indices0 = build_orig_indices(
            mode=mode,
            M=int(M0),        # current sensors count
            W=int(W0),        # current windows count
            M0=int(M0),
            W0=int(W0),
            active_features=active_features0,
            active_sensors=active_sensors0,
            active_windows=active_windows0,
        )
        input_dim0 = int(M0 * W0)

        baseline_selection = np.ones(M0, dtype=int)
        selected_original = active_sensors0.copy()

        baseline_importance, _ = compute_importance_sensors_from_mlp(
            hidden_Ws=baseline_hidden_Ws,
            M0=int(M0),
            W0=int(W0),
            active_sensors=active_sensors0,
        )

    elif mode == TRUST_MODE_WINDOWS:
        orig_indices0 = build_orig_indices(
            mode=mode,
            M=int(M0),
            W=int(W0),
            M0=int(M0),
            W0=int(W0),
            active_features=active_features0,
            active_sensors=active_sensors0,
            active_windows=active_windows0,
        )
        input_dim0 = int(M0 * W0)

        baseline_selection = np.ones(W0, dtype=int)
        selected_original = active_windows0.copy()

        baseline_importance, _ = compute_importance_windows_from_mlp(
            hidden_Ws=baseline_hidden_Ws,
            M0=int(M0),
            W0=int(W0),
            active_windows=active_windows0,
        )

    else:
        raise ValueError(f"Invalid mode={mode}")

    # Normalize importance safely
    baseline_imp_masked = np.asarray(baseline_importance, dtype=float).copy()
    total = float(baseline_imp_masked.sum())
    baseline_imp_norm = (baseline_imp_masked / total) if total > 0 else baseline_imp_masked.copy()

    baseline_iter_info = {
        "iter_idx": 0,
        "n0_value": int(max_n0),   # IMPORTANT: max_n0 depends on MODE (M0*W0 or M0 or W0)
        "mode": str(mode),

        "is_baseline": True,
        "is_final_mlp": False,

        # For TRANSFER alignment and reproducibility:
        # orig_indices is always local_flat -> orig_flat (length == input_dim0)
        "orig_indices": np.asarray(orig_indices0, dtype=int),
        "input_dim": int(input_dim0),

        # Selection/importance vectors must match MODE dimension
        "selection_vector": np.asarray(baseline_selection, dtype=int),
        "selected_original": np.asarray(selected_original, dtype=int),

        "importance_vector": np.asarray(baseline_imp_masked, dtype=float),
        "importance_vector_norm": np.asarray(baseline_imp_norm, dtype=float),

        "hidden_sizes": list(cfg.hidden_layers),
        "mlp_mode": "baseline",
        "mlp_metrics": dict(baseline_metrics),

        "baseline_mae_clusters": baseline_mae_clusters,
        "baseline_mae_floor": baseline_mae_floor,
    }

    _save_iteration_checkpoint(out_dir, iter_idx=0, iter_info=baseline_iter_info)

    # Training config for surrogate MLP
    mlp_cfg = TrustMLPTrainConfig(
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
    )

    for iter_idx, n0_value in enumerate(range(start_n0, 0, -1), start=start_iter_idx):
        if verbose:
            print(f"N0={n0_value}")

        # ------------------------------------------------------------
        # 1) Flatten current datasets (for surrogate MLP)
        # ------------------------------------------------------------
        X_train_flat, X_val_flat, ds_info = _flatten_train_val(X_train, X_val)
        M = int(ds_info["M"])
        W = int(ds_info["W"])
        input_dim = int(ds_info["input_dim"])

        # ------------------------------------------------------------
        # 2) Local -> original mapping for current space
        # ------------------------------------------------------------
        orig_indices = build_orig_indices(
            mode=mode,
            M=M,
            W=W,
            M0=M0,
            W0=W0,
            active_features=active_features,
            active_sensors=active_sensors,
            active_windows=active_windows,
        )

        # ------------------------------------------------------------
        # 3) Train surrogate MLP (REBUILD / TRANSFER)
        # ------------------------------------------------------------
        # Important: "baseline as surrogate" is only valid on a FRESH run at N0=max_n0.
        use_baseline_surrogate = (
            (not resume or start_iter_idx == 1) and
            (n0_value == max_n0 - 1) and
            (baseline_weights_list is not None) and
            (baseline_metrics is not None)
        )

        if use_baseline_surrogate:
            # Use baseline artifacts as the surrogate for N0=max
            model_info = {"final_metrics": dict(baseline_metrics)}
            hidden_sizes = list(cfg.hidden_layers)
            hidden_Ws = baseline_hidden_Ws
            hidden_bs = baseline_hidden_bs

            prev_model_weights_list = list(baseline_weights_list)
            prev_orig_indices = orig_indices.copy()

            mlp_train_time = 0.0  # No training performed

            if verbose:
                print("MLP surrogate: USING BASELINE MODEL")
                pretty_mlp_progress(n0_value, model_info["final_metrics"])

        else:
            t0 = time.perf_counter()

            if (
                mlp_mode == MLP_TRANSFER
                and prev_model_weights_list is not None
                and prev_orig_indices is not None
            ):
                model_info = train_and_collect_transfer(
                    build_trust_mlp_fn=build_trust_mlp,
                    input_dim=input_dim,
                    hidden_sizes=list(cfg.hidden_layers),
                    X_train=X_train_flat,
                    y_train=y_train,
                    X_val=X_val_flat,
                    y_val=y_val,
                    prev_weights_list=prev_model_weights_list,
                    prev_orig_indices=prev_orig_indices,
                    curr_orig_indices=orig_indices,
                    cfg=mlp_cfg,
                )
            else:
                model_info = train_and_collect(
                    build_trust_mlp_fn=build_trust_mlp,
                    input_dim=input_dim,
                    hidden_sizes=list(cfg.hidden_layers),
                    X_train=X_train_flat,
                    y_train=y_train,
                    X_val=X_val_flat,
                    y_val=y_val,
                    cfg=mlp_cfg,
                )

            mlp_train_time = float(time.perf_counter() - t0)

            if verbose:
                pretty_mlp_progress(n0_value, model_info["final_metrics"])

            prev_model_weights_list = list(model_info["model"].get_weights())
            prev_orig_indices = orig_indices.copy()

            _save_mlp_weights_checkpoint(out_dir, iter_idx=iter_idx, weights_list=prev_model_weights_list)

            hidden_sizes, hidden_Ws, hidden_bs = extract_mlp_weights(model_info)

        # ------------------------------------------------------------
        # 4) Project fixed centroids into current local space
        # ------------------------------------------------------------
        centroids_mw_local = _project_centroids_full_to_local(
            centroids_mw_full=centroids_full,
            mode=mode,
            active_features=active_features,
            active_sensors=active_sensors,
            active_windows=active_windows,
        )

        # ------------------------------------------------------------
        # 5) Solve MILP-1 (no cap)
        # ------------------------------------------------------------
        model_pack_1 = full_variable_layer_trust(
            n0_value=int(n0_value),
            centroids_mw=centroids_mw_local,
            y_c=y_c,
            alpha=alpha,
            hidden_lengths=hidden_sizes,
            hidden_Ws=hidden_Ws,
            hidden_bs=hidden_bs,
            mode=mode,
            beta=float(beta)*10,  # MILP-1 uses a higher beta to encourage sparsity
            milp_mode=MILP_MODE_1,
            baseline_mode=BASELINE_MODE_NONE,
            baseline_mae=None,
            baseline_slack=0.0,
        )

        summary_1 = model_pack_1["model"].solve(
            solver="gurobi",
            options=Options(
                threads=multiprocessing.cpu_count(),
                enable_scaling=True,
                try_partial_integer_solution=True,
            ),
        )
        if verbose:
            pretty_milp_df("MILP-1 Summary", summary_1)

        # ------------------------------------------------------------
        # 6) Solve MILP-2 (optional cap) + warm start
        # ------------------------------------------------------------
        model_pack_2 = full_variable_layer_trust(
            n0_value=int(n0_value),
            centroids_mw=centroids_mw_local,
            y_c=y_c,
            alpha=alpha,
            hidden_lengths=hidden_sizes,
            hidden_Ws=hidden_Ws,
            hidden_bs=hidden_bs,
            mode=mode,
            beta=float(beta),
            milp_mode=MILP_MODE_2,
            baseline_mode=baseline_mode,
            baseline_mae=baseline_mae_floor if baseline_mode == BASELINE_MODE_CAPPED else None,
            baseline_slack=0.0,
        )

        apply_warm_start(model_pack_2, model_pack_1)

        with suppress_gams_warnings():
            summary_2 = model_pack_2["model"].solve(
                solver="gurobi",
                options=Options(
                    threads=multiprocessing.cpu_count(),
                    enable_scaling=True,
                    try_partial_integer_solution=True,
                    time_limit=int(milp_time_cap),
                ),
            )

        if verbose:
            pretty_milp_df("MILP-2 Summary", summary_2)

        # ------------------------------------------------------------
        # 7) Extract selection and importance (mode-specific)
        # ------------------------------------------------------------
        if mode == TRUST_MODE_FULL:
            use_milp2 = True
            if summary_2 is None or len(summary_2) == 0:
                use_milp2 = False
            else:
                row2 = summary_2.iloc[0]
                if str(row2.get("Model Status", "")).strip() == "NoSolutionReturned":
                    use_milp2 = False

            if use_milp2:
                importance_vector, _ = compute_importance_full_milp2(
                    model_pack_2=model_pack_2,
                    M0=M0,
                    W0=W0,
                    active_features=active_features,
                )
                z0_pack = model_pack_2.get("Z0")
            else:
                importance_vector, _ = compute_importance_full_from_mlp(
                    hidden_Ws=hidden_Ws,
                    M0=M0,
                    W0=W0,
                    active_features=active_features,
                )
                z0_pack = model_pack_1.get("Z0")

            if z0_pack is None:
                raise RuntimeError("Z0 is missing for selection extraction.")

            sel_df = z0_pack.records.copy()
            selected_local = sel_df.loc[sel_df["level"] > 0.5, "m"].astype(int).values - 1
            selected_original = active_features[selected_local]

            selection_vector = np.zeros(M0 * W0, dtype=int)
            selection_vector[selected_original] = 1

            importance_masked = np.asarray(importance_vector, dtype=float).copy()
            importance_masked[selection_vector == 0] = 0.0
            total = float(importance_masked.sum())
            importance_vector_norm = (importance_masked / total) if total > 0 else importance_masked.copy()
            importance_vector = importance_masked

            active_features = selected_original

            mask_local = np.zeros(M * W, dtype=bool)
            mask_local[selected_local] = True
            mask_local_mw = mask_local.reshape(M, W)

            X_train = X_train[:, mask_local_mw].reshape(X_train.shape[0], -1, 1)
            X_val = X_val[:, mask_local_mw].reshape(X_val.shape[0], -1, 1)

            selected_sensors = np.unique(selected_original // W0) + 1
            selected_windows = np.unique(selected_original % W0) + 1

        elif mode == TRUST_MODE_SENSORS:
            importance_vector, _ = compute_importance_sensors_milp2(
                model_pack_2=model_pack_2,
                M0=M0,
                W0=W0,
                active_sensors=active_sensors,
            )

            sel_df = model_pack_2["S0"].records.copy()
            selected_local = sel_df.loc[sel_df["level"] > 0.5, "s"].astype(int).values - 1
            selected_original = active_sensors[selected_local]

            selection_vector = np.zeros(M0, dtype=int)
            selection_vector[selected_original] = 1

            importance_masked = np.asarray(importance_vector, dtype=float).copy()
            importance_masked[selection_vector == 0] = 0.0
            total = float(importance_masked.sum())
            importance_vector_norm = (importance_masked / total) if total > 0 else importance_masked.copy()
            importance_vector = importance_masked

            active_sensors = selected_original

            X_train = X_train[:, selected_local, :]
            X_val = X_val[:, selected_local, :]

            selected_sensors = selected_original + 1
            selected_windows = active_windows + 1

        else:  # TRUST_MODE_WINDOWS
            importance_vector, _ = compute_importance_windows_milp2(
                model_pack_2=model_pack_2,
                M0=M0,
                W0=W0,
                active_windows=active_windows,
            )

            sel_df = model_pack_2["W0"].records.copy()
            selected_local = sel_df.loc[sel_df["level"] > 0.5, "w"].astype(int).values - 1
            selected_original = active_windows[selected_local]

            selection_vector = np.zeros(W0, dtype=int)
            selection_vector[selected_original] = 1

            importance_masked = np.asarray(importance_vector, dtype=float).copy()
            importance_masked[selection_vector == 0] = 0.0
            total = float(importance_masked.sum())
            importance_vector_norm = (importance_masked / total) if total > 0 else importance_masked.copy()
            importance_vector = importance_masked

            active_windows = selected_original

            X_train = X_train[:, :, selected_local]
            X_val = X_val[:, :, selected_local]

            selected_windows = selected_original + 1
            selected_sensors = active_sensors + 1

        if verbose:
            pretty_selection_progress(
                n0_value=n0_value,
                selection_vector=selection_vector,
                selected_sensors=selected_sensors,
                selected_windows=selected_windows,
            )

        # ------------------------------------------------------------
        # 8) Save checkpoint
        # ------------------------------------------------------------
        iter_info: Dict[str, Any] = {
            "iter_idx": int(iter_idx),
            "n0_value": int(n0_value),
            "mode": str(mode),

            "baseline_mode": str(baseline_mode),
            "baseline_slack": float(baseline_slack),
            "baseline_mae_clusters": None if baseline_mae_clusters is None else float(baseline_mae_clusters),
            "baseline_mae_floor": None if baseline_mae_floor is None else float(baseline_mae_floor),

            "summary_1": summary_1,
            "summary_2": summary_2,

            "selection_df": sel_df,
            "selected_original": np.asarray(selected_original, dtype=int),
            "selection_vector": np.asarray(selection_vector, dtype=int),
            "importance_vector": np.asarray(importance_vector, dtype=float),
            "importance_vector_norm": np.asarray(importance_vector_norm, dtype=float),

            "selected_sensors": np.asarray(selected_sensors, dtype=int),
            "selected_windows": np.asarray(selected_windows, dtype=int),

            "X_train_shape_next": tuple(int(x) for x in X_train.shape),
            "X_val_shape_next": tuple(int(x) for x in X_val.shape),

            "M_curr": int(M),
            "W_curr": int(W),
            "input_dim": int(input_dim),
            "orig_indices": np.asarray(orig_indices, dtype=int),

            "surrogate_from_baseline": bool(use_baseline_surrogate),

            "hidden_sizes": list(hidden_sizes),
            "mlp_mode": str(mlp_mode),
            "mlp_metrics": dict(model_info["final_metrics"]),
            "mlp_train_time": float(mlp_train_time),
        }

        _save_iteration_checkpoint(out_dir, iter_idx=iter_idx, iter_info=iter_info)

        # Stop criteria
        if mode == TRUST_MODE_SENSORS and X_train.shape[1] <= 1:
            if verbose:
                print("Stopping: only 1 sensor left.")
            break
        if mode == TRUST_MODE_WINDOWS and X_train.shape[2] <= 1:
            if verbose:
                print("Stopping: only 1 window left.")
            break
        if mode == TRUST_MODE_FULL and X_train.shape[1] <= 1:
            if verbose:
                print("Stopping: only 1 feature left.")
            break
    
    # -------------------------
    # Final MLP (n0_value = 0)
    # -------------------------
    final_iter_idx = int(_find_last_iteration(out_dir) or 0) + 1

    X_train_flat, X_val_flat, ds_info = _flatten_train_val(X_train, X_val) 
    input_dim_final = int(ds_info["input_dim"])

    orig_indices_final = build_orig_indices(
        mode=mode,
        M=int(ds_info["M"]),
        W=int(ds_info["W"]),
        M0=M0,
        W0=W0,
        active_features=active_features,
        active_sensors=active_sensors,
        active_windows=active_windows,
    )

    t0 = time.perf_counter()

    if mlp_mode == MLP_TRANSFER and prev_model_weights_list is not None and prev_orig_indices is not None:
        final_info = train_and_collect_transfer(
            build_trust_mlp_fn=build_trust_mlp,
            input_dim=input_dim_final,
            hidden_sizes=list(cfg.hidden_layers),
            X_train=X_train_flat,
            y_train=y_train,
            X_val=X_val_flat,
            y_val=y_val,
            prev_weights_list=prev_model_weights_list,
            prev_orig_indices=prev_orig_indices,
            curr_orig_indices=orig_indices_final,
            cfg=mlp_cfg,
        )
    else:
        final_info = train_and_collect(
            build_trust_mlp_fn=build_trust_mlp,
            input_dim=input_dim_final,
            hidden_sizes=list(cfg.hidden_layers),
            X_train=X_train_flat,
            y_train=y_train,
            X_val=X_val_flat,
            y_val=y_val,
            cfg=mlp_cfg,
        )

    final_train_time = float(time.perf_counter() - t0)


    final_iter_info = {
        "iter_idx": int(final_iter_idx),
        "n0_value": 0,
        "mode": str(mode),
        "is_final_mlp": True,
        "orig_indices": np.asarray(orig_indices_final, dtype=int),
        "input_dim": int(input_dim_final),
        "hidden_sizes": list(cfg.hidden_layers),
        "mlp_mode": str(mlp_mode),
        "mlp_metrics": dict(final_info["final_metrics"]),
        "mlp_train_time": float(final_train_time),
    }

    _save_mlp_weights_checkpoint(out_dir, iter_idx=final_iter_idx, weights_list=list(final_info["model"].get_weights()))
    _save_iteration_checkpoint(out_dir, iter_idx=final_iter_idx, iter_info=final_iter_info)

    history_full = _load_all_iterations(out_dir)

    finalize_trust_run_artifacts(Path(out_dir))

    result = {
        "out_dir": str(out_dir),
        "run_config": asdict(cfg),
        "history": history_full,
    }

    with open(out_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)

    return result


def run_trust_with_baseline_training(
    *,
    dataset_name: str,
    window_tag: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    results_root: Optional[Path] = None,
    hidden_layers: Sequence[int] = (10,),
    mode: str = TRUST_MODE_SENSORS,
    C: int = 50,
    beta: float = 0.5,
    mlp_mode: str = MLP_REBUILD,
    milp_time_cap: int = 60,
    learning_rate: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
    baseline_mode: str = BASELINE_MODE_NONE,
    baseline_slack: float = 0.0,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end helper for publication demos:
    1) Trains a baseline MLP on the full SxW feature grid.
    2) Extracts baseline weights/metrics.
    3) Runs the iterative TRUST pipeline.
    """
    X_train_flat, X_val_flat, ds_info = _flatten_train_val(X_train, X_val)

    baseline_info = train_and_collect(
        build_trust_mlp_fn=build_trust_mlp,
        input_dim=int(ds_info["input_dim"]),
        hidden_sizes=list(hidden_layers),
        X_train=X_train_flat,
        y_train=y_train,
        X_val=X_val_flat,
        y_val=y_val,
        cfg=TrustMLPTrainConfig(
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=0,
        ),
    )
    _, baseline_hidden_Ws, baseline_hidden_bs = extract_mlp_weights(baseline_info)

    return execute_trust_algorithm(
        dataset_name=dataset_name,
        window_tag=window_tag,
        results_root=results_root,
        hidden_layers=hidden_layers,
        mode=mode,
        C=C,
        beta=beta,
        mlp_mode=mlp_mode,
        milp_time_cap=milp_time_cap,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        baseline_mode=baseline_mode,
        baseline_slack=baseline_slack,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        baseline_hidden_Ws=baseline_hidden_Ws,
        baseline_hidden_bs=baseline_hidden_bs,
        baseline_metrics=dict(baseline_info["final_metrics"]),
        baseline_weights_list=list(baseline_info["model"].get_weights()),
        seed=seed,
        verbose=verbose,
        resume=resume,
    )