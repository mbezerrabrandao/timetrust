from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np


# =========================
# Constants / Modes
# =========================

TRUST_MODE_FULL = "full"
TRUST_MODE_SENSORS = "sensors"
TRUST_MODE_WINDOWS = "windows"

TRUST_MODES_DESCRIPTION: Dict[str, str] = {
    TRUST_MODE_FULL: "Selection of features across all lags and sensors",
    TRUST_MODE_SENSORS: "Selection of features across all lags per sensor",
    TRUST_MODE_WINDOWS: "Selection of features across all sensors per window",
}

MILP_MODE_1 = "MILP-1"
MILP_MODE_2 = "MILP-2"

MILP_MODES_DESCRIPTION: Dict[str, str] = {
    MILP_MODE_1: "First layer weights are fixed parameters",
    MILP_MODE_2: "First layer weights become decision variables",
}

MLP_REBUILD = "mlp_rebuild"
MLP_TRANSFER = "mlp_transfer"

MLP_MODES_DESCRIPTION: Dict[str, str] = {
    MLP_REBUILD: "Rebuilds the MLP on each selection",
    MLP_TRANSFER: "Retrains / fine-tunes after selection (transfer-style update)",
}

BASELINE_MODE_NONE = "none"
BASELINE_MODE_CAPPED = "capped"

BASELINE_MODES_DESCRIPTION: Dict[str, str] = {
    BASELINE_MODE_NONE: "No baseline limitation",
    BASELINE_MODE_CAPPED: "Capped to baseline MLP performance",
}


# =========================
# Bounds (ReLU pre-activation z)
# =========================

def compute_weight_bounds_first_layer(
    W1: np.ndarray,
    b1: np.ndarray,
    factor: float = 2.0,
    eps_w: float = 1e-6,
    max_w: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds symmetric bounds around zero for first-layer weights/biases based on their magnitudes:
        LB = -factor * max(|param|, eps_w)
        UB =  factor * max(|param|, eps_w)

    Optionally clamps magnitudes by max_w for numerical safety.
    """
    W1 = np.asarray(W1)
    b1 = np.asarray(b1).reshape(-1)

    mag_W = np.abs(W1)
    mag_b = np.abs(b1)

    if max_w is not None:
        mag_W = np.minimum(mag_W, max_w)
        mag_b = np.minimum(mag_b, max_w)

    mag_W = np.maximum(mag_W, eps_w)
    mag_b = np.maximum(mag_b, eps_w)

    LB_W1 = -factor * mag_W
    UB_W1 =  factor * mag_W
    LB_b1 = -factor * mag_b
    UB_b1 =  factor * mag_b

    return LB_W1, UB_W1, LB_b1, UB_b1


def compute_relu_bounds_all_layers(
    centroids_flat: np.ndarray,
    hidden_Ws: List[np.ndarray],
    hidden_bs: List[np.ndarray],
    beta: float = 0.3,
    eps: float = 1e-6,
    strict_zero_crossing: bool = False,
    warn_threshold: float = 0.5,
    # Optional bounds for first layer (MILP-2)
    LB_W1: Optional[np.ndarray] = None,
    UB_W1: Optional[np.ndarray] = None,
    LB_b1: Optional[np.ndarray] = None,
    UB_b1: Optional[np.ndarray] = None,
    use_interval_first_layer: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Compute empirical lower/upper bounds (LB, UB) for each layer's pre-activation z.

    - Bounds are estimated from z-values evaluated at centroids (centroids_flat).
    - For layer 1, if use_interval_first_layer=True and (LB_W1, UB_W1, LB_b1, UB_b1)
      are provided, bounds are computed via interval arithmetic using input ranges and
      parameter ranges (MILP-2 case).

    Returns
    -------
    LBs : list[np.ndarray]
        LB per hidden layer (output layer excluded).
    UBs : list[np.ndarray]
        UB per hidden layer (output layer excluded).
    z_vals_layers : list[np.ndarray]
        z-values per hidden layer evaluated at centroids (for debugging/analysis).
    """
    if len(hidden_Ws) != len(hidden_bs):
        raise ValueError("hidden_Ws and hidden_bs must have same length.")

    # Last element corresponds to output layer (linear), excluded from ReLU bounds
    n_hidden_layers = len(hidden_Ws) - 1
    if n_hidden_layers <= 0:
        raise ValueError("Expected at least 1 hidden layer + 1 output layer.")

    if beta < 0:
        raise ValueError(f"beta={beta} cannot be negative.")
    if eps <= 0:
        raise ValueError(f"eps must be > 0. Got eps={eps}")
    if not (0.0 <= warn_threshold <= 1.0):
        raise ValueError("warn_threshold must be a fraction in [0, 1].")

    a = np.asarray(centroids_flat)
    if a.ndim != 2:
        raise ValueError(f"centroids_flat must be 2D (C, input_dim), got {a.shape}")
    if np.isnan(a).any() or np.isinf(a).any():
        raise ValueError("centroids_flat contains NaN/Inf values.")

    LBs: List[np.ndarray] = []
    UBs: List[np.ndarray] = []
    z_vals_layers: List[np.ndarray] = []

    # Input range (for interval arithmetic on layer 1)
    a_min = a.min(axis=0)
    a_max = a.max(axis=0)

    for l in range(n_hidden_layers):
        W_l = np.asarray(hidden_Ws[l])
        b_l = np.asarray(hidden_bs[l]).reshape(-1)

        if a.shape[1] != W_l.shape[0]:
            raise ValueError(
                f"Layer {l+1}: shape mismatch: a.shape={a.shape}, W_l.shape={W_l.shape}"
            )
        if W_l.shape[1] != b_l.shape[0]:
            raise ValueError(
                f"Layer {l+1}: bias shape mismatch: W_l outputs={W_l.shape[1]} "
                f"but b_l.shape={b_l.shape}"
            )

        # ----------------------------------------------
        # Layer 1 interval arithmetic (MILP-2 support)
        # ----------------------------------------------
        if (
            l == 0
            and use_interval_first_layer
            and (LB_W1 is not None)
            and (UB_W1 is not None)
            and (LB_b1 is not None)
            and (UB_b1 is not None)
        ):
            LB_W1_arr = np.asarray(LB_W1)
            UB_W1_arr = np.asarray(UB_W1)
            LB_b1_arr = np.asarray(LB_b1).reshape(-1)
            UB_b1_arr = np.asarray(UB_b1).reshape(-1)

            if LB_W1_arr.shape != W_l.shape or UB_W1_arr.shape != W_l.shape:
                raise ValueError(
                    f"Layer 1: LB_W1/UB_W1 shape mismatch: "
                    f"W_l.shape={W_l.shape}, LB_W1.shape={LB_W1_arr.shape}, "
                    f"UB_W1.shape={UB_W1_arr.shape}"
                )
            if LB_b1_arr.shape != b_l.shape or UB_b1_arr.shape != b_l.shape:
                raise ValueError(
                    f"Layer 1: LB_b1/UB_b1 shape mismatch: "
                    f"b_l.shape={b_l.shape}, LB_b1.shape={LB_b1_arr.shape}, "
                    f"UB_b1.shape={UB_b1_arr.shape}"
                )

            a_min_col = a_min[:, None]
            a_max_col = a_max[:, None]

            prod1 = a_min_col * LB_W1_arr
            prod2 = a_min_col * UB_W1_arr
            prod3 = a_max_col * LB_W1_arr
            prod4 = a_max_col * UB_W1_arr

            z_min_terms = np.minimum.reduce([prod1, prod2, prod3, prod4])
            z_max_terms = np.maximum.reduce([prod1, prod2, prod3, prod4])

            z_min = z_min_terms.sum(axis=0) + LB_b1_arr
            z_max = z_max_terms.sum(axis=0) + UB_b1_arr

            if np.any(z_max < z_min):
                raise ValueError("Layer 1: interval arithmetic produced z_max < z_min.")

            range_z = z_max - z_min
            if np.any(range_z == 0):
                print("[Warning] Layer 1: zero range (z_min=z_max) for some neurons (interval).")

            margin = beta * range_z + eps
            if np.any(margin <= 0):
                raise ValueError(f"Layer 1: non-positive margin. min(margin)={margin.min()}")

            LB_l = z_min - margin
            UB_l = z_max + margin

            # Nominal z (only for debug/analysis)
            z = a @ W_l + b_l

        else:
            # Empirical bounds from centroids (original approach)
            z = a @ W_l + b_l
            if np.isnan(z).any() or np.isinf(z).any():
                raise ValueError(f"Layer {l+1}: z contains NaN/Inf values.")

            z_min = z.min(axis=0)
            z_max = z.max(axis=0)
            range_z = z_max - z_min

            if np.any(range_z < 0):
                raise ValueError(f"Layer {l+1}: z_max < z_min for some neurons.")
            if np.any(range_z == 0):
                print(f"[Warning] Layer {l+1}: zero range (z_min=z_max) for some neurons.")

            margin = beta * range_z + eps
            if np.any(margin <= 0):
                raise ValueError(
                    f"Layer {l+1}: non-positive margin. min(margin)={margin.min()}"
                )

            LB_l = z_min - margin
            UB_l = z_max + margin

        if np.any(LB_l >= UB_l):
            raise ValueError(
                f"Layer {l+1}: invalid bounds (LB>=UB) for some neurons."
            )

        # Zero-crossing sanity check: LB < 0 < UB is ideal for "tight" ReLU big-M
        violates = (LB_l >= 0) | (UB_l <= 0)
        frac_viol = float(violates.mean())

        if frac_viol > 0:
            msg = (
                f"Layer {l+1}: {frac_viol*100:.1f}% neurons violate LB<0<UB "
                f"(always active/inactive). "
                f"LB>=0: {(LB_l>=0).sum()}, UB<=0: {(UB_l<=0).sum()}."
            )
            if strict_zero_crossing:
                raise ValueError(msg)
            if frac_viol >= warn_threshold:
                print("[Warning]", msg)

        LBs.append(LB_l)
        UBs.append(UB_l)
        z_vals_layers.append(z)

        # Propagate activations using nominal z
        a = np.maximum(0.0, z)
        if np.all(a == 0):
            print(f"[Warning] Layer {l+1}: all activations are zero after ReLU.")

    return LBs, UBs, z_vals_layers