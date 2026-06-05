# trust/pipeline_artifacts.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Iteration IO helpers
# -----------------------------

_ITER_JSON_RE = re.compile(r"^iter_(\d{3})\.json$")
_ITER_NPZ_RE = re.compile(r"^iter_(\d{3})\.npz$")


def _iter_indices(out_dir: Path) -> List[int]:
    out_dir = Path(out_dir)
    iters: List[int] = []
    for p in out_dir.glob("iter_*.json"):
        m = _ITER_JSON_RE.match(p.name)
        if m:
            iters.append(int(m.group(1)))
    return sorted(set(iters))


def _load_iter_json(out_dir: Path, iter_idx: int) -> Dict[str, Any]:
    p = Path(out_dir) / f"iter_{iter_idx:03d}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing iteration json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_iter_npz(out_dir: Path, iter_idx: int) -> Dict[str, np.ndarray]:
    p = Path(out_dir) / f"iter_{iter_idx:03d}.npz"
    if not p.exists():
        return {}
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}


# -----------------------------
# Robust MILP time extraction
# -----------------------------

def _first_row(summary_obj: Any) -> Optional[Dict[str, Any]]:
    """
    summary_obj is expected to be either:
      - a dict with key "summary_1"/"summary_2" already converted to list-of-dicts, or
      - a list-of-dicts, or
      - a pandas-like object already serialized.
    We return the first row as a dict, if possible.
    """
    if summary_obj is None:
        return None
    if isinstance(summary_obj, list) and len(summary_obj) > 0 and isinstance(summary_obj[0], dict):
        return summary_obj[0]
    if isinstance(summary_obj, dict):
        # Already a single row dict
        return summary_obj
    return None


def _extract_solve_time_seconds(summary_row: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Extract solver time in seconds from GAMSPy summary row.
    """
    if not summary_row:
        return None

    v = summary_row.get("Solver Time", None)
    if v is None:
        return None

    try:
        return float(v)
    except Exception:
        return None


def _extract_status(summary_row: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extract model/solver status.
    """
    if not summary_row:
        return None

    model_status = summary_row.get("Model Status", None)
    solver_status = summary_row.get("Solver Status", None)

    if model_status is not None:
        return str(model_status)

    if solver_status is not None:
        return str(solver_status)

    return None


def _extract_objective(summary_row: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Extract objective value.
    """
    if not summary_row:
        return None

    v = summary_row.get("Objective", None)
    if v is None:
        return None

    try:
        return float(v)
    except Exception:
        return None

def save_vectors_selection_importance(
    out_dir: Path,
    data: Dict[str, np.ndarray],
    npz_name: str = "vectors_selection_importance.npz",
    json_name: str = "vectors_selection_importance.json",
) -> None:
    """
    Save selection/importance vectors in both NPZ and JSON formats.

    Parameters
    ----------
    out_dir : Path
        Experiment output folder.
    data : dict
        Keys -> arrays (selection, importance, n0, iters, etc).
    npz_name : str
        Output NPZ filename.
    json_name : str
        Output JSON filename.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Save NPZ (fast/compact)
    # -----------------------
    np.savez_compressed(out_dir / npz_name, **data)

    # -----------------------
    # Save JSON (portable)
    # -----------------------
    json_payload: Dict[str, Any] = {}

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            json_payload[k] = v.tolist()
        else:
            json_payload[k] = v

    with open(out_dir / json_name, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

# -----------------------------
# Main finalize function
# -----------------------------

def finalize_trust_run_artifacts(
    out_dir: Path,
    *,
    vectors_npz_name: str = "vectors_selection_importance.npz",
    mlp_metrics_csv_name: str = "mlp_metrics_by_nfeatures.csv",
    milp_summaries_csv_name: str = "milp_summaries_iter1_to_penultimate.csv",
    timing_json_name: str = "experiment_total_time.json",
) -> Dict[str, str]:
    """
    Creates 3 (plus 1 extra) files in the results folder:

    1) experiment_total_time.json:
       total seconds = sum over iterations of (mlp_train_time + milp1_solve_time + milp2_solve_time)

    2) vectors_selection_importance.npz:
       selection_vector and importance_vector_norm from iter 0 to the penultimate iter
       (excludes final MLP record with n0_value=0 or is_final_mlp=True)

    3) mlp_metrics_by_nfeatures.csv:
       MLP metrics per iteration from iter 1 to the last iter (includes final MLP if present)
       Assumes iter 1 metrics are baseline metrics (as you requested).

    Extra) milp_summaries_iter1_to_penultimate.csv:
       MILP-1 and MILP-2 key fields from iter 1 to penultimate iter.

    Returns a dict with created file paths.
    """
    out_dir = Path(out_dir)
    iters = _iter_indices(out_dir)
    if not iters:
        raise RuntimeError(f"No iter_*.json found under: {out_dir}")

    # Load all meta first
    metas: List[Tuple[int, Dict[str, Any]]] = [(i, _load_iter_json(out_dir, i)) for i in iters]

    # Identify final iter (final MLP) and penultimate iter (last trust iter)
    final_iter_idx: Optional[int] = None
    for i, meta in metas:
        if bool(meta.get("is_final_mlp", False)) or int(meta.get("n0_value", -999)) == 0:
            final_iter_idx = i

    penultimate_iter_idx: int
    if final_iter_idx is None:
        # No final MLP marker, then penultimate is simply the last iter
        penultimate_iter_idx = iters[-1]
    else:
        # Penultimate is the iteration right before the final MLP record
        prev_candidates = [i for i in iters if i < final_iter_idx]
        if not prev_candidates:
            raise RuntimeError("Final MLP exists but there is no previous iteration.")
        penultimate_iter_idx = prev_candidates[-1]

    # -----------------------------
    # 1) Total execution time
    # -----------------------------
    total_mlp = 0.0
    total_milp1 = 0.0
    total_milp2 = 0.0

    per_iter_times: List[Dict[str, Any]] = []

    for i, meta in metas:
        mlp_t = meta.get("mlp_train_time", 0.0)
        try:
            mlp_t = float(mlp_t) if mlp_t is not None else 0.0
        except Exception:
            mlp_t = 0.0

        s1_row = _first_row(meta.get("summary_1", None))
        s2_row = _first_row(meta.get("summary_2", None))

        t1 = _extract_solve_time_seconds(s1_row) or 0.0
        t2 = _extract_solve_time_seconds(s2_row) or 0.0

        total_mlp += mlp_t
        total_milp1 += float(t1)
        total_milp2 += float(t2)

        per_iter_times.append(
            {
                "iter_idx": int(i),
                "n0_value": int(meta.get("n0_value", -1)),
                "mlp_train_time_s": float(mlp_t),
                "milp1_solve_time_s": float(t1),
                "milp2_solve_time_s": float(t2),
                "is_final_mlp": bool(meta.get("is_final_mlp", False)),
                "is_baseline": bool(meta.get("surrogate_from_baseline", False)),
            }
        )

    timing_payload = {
        "total_time_s": float(total_mlp + total_milp1 + total_milp2),
        "mlp_train_time_s": float(total_mlp),
        "milp1_solve_time_s": float(total_milp1),
        "milp2_solve_time_s": float(total_milp2),
        "per_iteration": per_iter_times,
    }

    timing_path = out_dir / timing_json_name
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2)

    # -----------------------------
    # 2) Vectors (iter 0 to penultimate)
    # -----------------------------
    vec_iters = [i for i in iters if i <= penultimate_iter_idx]

    sel_list: List[np.ndarray] = []
    imp_list: List[np.ndarray] = []
    n0_list: List[int] = []
    iter_list: List[int] = []

    for i in vec_iters:
        meta = _load_iter_json(out_dir, i)

        # Skip final MLP record (n0_value=0) if present
        if bool(meta.get("is_final_mlp", False)) or int(meta.get("n0_value", -999)) == 0:
            continue

        arrs = _load_iter_npz(out_dir, i)

        # Require selection vector
        if "selection_vector" not in arrs:
            continue

        # Importance:
        if "importance_vector" in arrs:
            importance_key = "importance_vector"
        else:
            continue

        selection_vec = np.asarray(arrs["selection_vector"], dtype=np.int8)
        importance_vec = np.asarray(arrs[importance_key], dtype=np.float32)

        sel_list.append(selection_vec)
        imp_list.append(importance_vec)
        n0_list.append(int(meta.get("n0_value", -1)))
        iter_list.append(int(i))

    # Build payload (safe even if empty)
    if len(sel_list) > 0:
        selection_matrix = np.stack(sel_list, axis=0)   # (K, D)
        importance_matrix = np.stack(imp_list, axis=0)  # (K, D)
    else:
        # Keep shapes consistent (0, D) is impossible without D, so store empty 1D
        selection_matrix = np.empty((0,), dtype=np.int8)
        importance_matrix = np.empty((0,), dtype=np.float32)

    payload = {
        "iter_idx": np.asarray(iter_list, dtype=np.int32),
        "n0_value": np.asarray(n0_list, dtype=np.int32),
        "selection_vector": selection_matrix,
        "importance_vector": importance_matrix,
    }

    save_vectors_selection_importance(
        out_dir=out_dir,
        data=payload,
        npz_name=vectors_npz_name,
        json_name="vectors_selection_importance.json",
    )



    # -----------------------------
    # 3) MLP metrics by number of features (iter 1 to last)
    # -----------------------------
    mlp_rows: List[Dict[str, Any]] = []
    for i, meta in metas:
        if int(i) < 1:
            continue

        metrics = meta.get("mlp_metrics", None)
        if not isinstance(metrics, dict):
            metrics = {}

        # Prefer explicit input_dim; otherwise infer from selection vector if available
        input_dim = meta.get("input_dim", None)
        n_features_active = None

        if input_dim is not None:
            try:
                n_features_active = int(input_dim)
            except Exception:
                n_features_active = None

        if n_features_active is None:
            arrs = _load_iter_npz(out_dir, i)
            if "selection_vector" in arrs:
                n_features_active = int(np.asarray(arrs["selection_vector"]).sum())

        row = {
            "iter_idx": int(i),
            "n0_value": int(meta.get("n0_value", -1)),
            "n_features": None if n_features_active is None else int(n_features_active),
            "is_final_mlp": bool(meta.get("is_final_mlp", False)),
            "is_baseline": bool(meta.get("surrogate_from_baseline", False)),
        }

        # Keep your surrogate metrics as MSE/RMSE/MAE/R2, per your decision
        for k in (
            "train_mse", "val_mse",
            "train_rmse", "val_rmse",
            "train_mae", "val_mae",
            "train_r2", "val_r2",
        ):
            v = metrics.get(k, None)
            row[k] = None if v is None else float(v)

        row["mlp_train_time_s"] = float(meta.get("mlp_train_time", 0.0) or 0.0)
        mlp_rows.append(row)

    mlp_df = pd.DataFrame(mlp_rows).sort_values(["iter_idx"])
    mlp_path = out_dir / mlp_metrics_csv_name
    mlp_df.to_csv(mlp_path, index=False)

    # -----------------------------
    # Extra) MILP summaries (iter 1 to penultimate)
    # -----------------------------
    milp_rows: List[Dict[str, Any]] = []
    for i in iters:
        if i < 1 or i > penultimate_iter_idx:
            continue

        meta = _load_iter_json(out_dir, i)
        if bool(meta.get("is_final_mlp", False)) or int(meta.get("n0_value", -999)) == 0:
            continue

        s1_row = _first_row(meta.get("summary_1", None))
        s2_row = _first_row(meta.get("summary_2", None))

        r = {
            "iter_idx": int(i),
            "n0_value": int(meta.get("n0_value", -1)),
            # MILP-1
            "milp1_status": _extract_status(s1_row),
            "milp1_objective": _extract_objective(s1_row),
            "milp1_solve_time_s": _extract_solve_time_seconds(s1_row),
            # MILP-2
            "milp2_status": _extract_status(s2_row),
            "milp2_objective": _extract_objective(s2_row),
            "milp2_solve_time_s": _extract_solve_time_seconds(s2_row),
        }
        milp_rows.append(r)

    milp_df = pd.DataFrame(milp_rows).sort_values(["iter_idx"])
    milp_path = out_dir / milp_summaries_csv_name
    milp_df.to_csv(milp_path, index=False)