from __future__ import annotations

from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

import gamspy as gp
from gamspy import Container, Set, Parameter, Variable, Equation, Sum, Model, Sense, Problem

from public_time_trust.utils.milp_utils import (
    TRUST_MODE_FULL,
    TRUST_MODE_SENSORS,
    TRUST_MODE_WINDOWS,
    TRUST_MODES_DESCRIPTION,
    MILP_MODE_1,
    MILP_MODE_2,
    MILP_MODES_DESCRIPTION,
    BASELINE_MODE_NONE,
    BASELINE_MODE_CAPPED,
    BASELINE_MODES_DESCRIPTION,
    compute_relu_bounds_all_layers,
    compute_weight_bounds_first_layer,
)


def full_variable_layer_trust(
    n0_value: int,
    centroids_mw: np.ndarray,
    y_c: np.ndarray,
    alpha: np.ndarray,
    hidden_lengths: List[int],
    hidden_Ws: List[np.ndarray],
    hidden_bs: List[np.ndarray],
    beta: float = 0.5,
    eps: float = 1e-3,
    mode: str = TRUST_MODE_FULL,
    milp_mode: str = MILP_MODE_1,
    baseline_mode: str = BASELINE_MODE_NONE,
    baseline_mae: float | None = None,
    baseline_slack: float = 0.0,
    w_factor: float = 2.0,
) -> Dict[str, Any]:
    """
    Build a TRUST-style MILP in GAMSPy.

    Parameters
    ----------
    n0_value
        Selection budget (N0) for chosen mode.
        - full: number of selected flattened features
        - sensors: number of selected sensors
        - windows: number of selected window positions (lags)
    centroids_mw
        Centroids with shape (C, M_sensors, W_window_len).
    y_c
        Representative target per centroid, shape (C,).
    alpha
        Cluster proportions, shape (C,).
    hidden_lengths
        Hidden layer sizes, e.g. [10], [10,10], [10,10,10].
    hidden_Ws, hidden_bs
        Network weights and biases as lists. Last entry is output layer.
        - hidden_Ws = [W1, W2, ..., W_out]
        - hidden_bs = [b1, b2, ..., b_out]
    mode
        TRUST selection mode: full, sensors, windows.
    milp_mode
        MILP-1 (fixed first layer) or MILP-2 (trainable first layer with linearization).
    w_factor
        MILP-2 bound scaling factor for first layer weights/biases.

    Returns
    -------
    dict with container, model, and key decision vars for solution extraction.
    """
    # --------------------
    # Mode validation
    # --------------------
    if mode not in TRUST_MODES_DESCRIPTION:
        raise ValueError(
            f"Invalid mode={mode}. Use one of: {', '.join(TRUST_MODES_DESCRIPTION.keys())}."
        )

    if milp_mode not in MILP_MODES_DESCRIPTION:
        raise ValueError(
            f"Invalid milp_mode={milp_mode}. Use {MILP_MODE_1} or {MILP_MODE_2}."
        )

    if baseline_mode not in BASELINE_MODES_DESCRIPTION:
        raise ValueError(
            f"Invalid baseline_mode={baseline_mode}. "
            f"Use one of: {', '.join(BASELINE_MODES_DESCRIPTION.keys())}."
        )

    if baseline_mode == BASELINE_MODE_CAPPED:
        if baseline_mae is None:
            raise ValueError(
                "baseline_mode='capped' requires baseline_mae to be provided."
            )
        if not (0.0 <= baseline_slack < 1.0):
            raise ValueError(
                f"baseline_slack must be in [0, 1), got {baseline_slack}")

    first_layer_trainable = (milp_mode == MILP_MODE_2)

    # Root object that will hold all sets/params/vars/equations
    cont = Container()

    # --------------------
    # Sets
    # --------------------
    C_clusters = int(y_c.shape[0])
    c = Set(cont, name="c", records=list(range(1, C_clusters + 1)), description="Cluster centres")

    M_features = int(centroids_mw.shape[1] * centroids_mw.shape[2])
    m = Set(cont, name="m", records=list(range(1, M_features + 1)), description="Features")

    # Define optional group sets
    S0 = None
    W0 = None

    if mode == TRUST_MODE_SENSORS:
        S_sensors = int(centroids_mw.shape[1])
        s = Set(cont, name="s", records=list(range(1, S_sensors + 1)), description="Sensors")
    elif mode == TRUST_MODE_WINDOWS:
        W_windows = int(centroids_mw.shape[2])
        w = Set(cont, name="w", records=list(range(1, W_windows + 1)), description="Windows")

    H_hiddens: List[int] = []
    hs: List[Set] = []
    for idx, hidden_length in enumerate(hidden_lengths):
        H_hidden = int(hidden_length)
        h = Set(
            cont,
            name=f"h{idx+1}",
            records=list(range(1, H_hidden + 1)),
            description=f"Hidden layer {idx+1} nodes",
        )
        H_hiddens.append(H_hidden)
        hs.append(h)

    # --------------------
    # Shape checks
    # --------------------
    C, M1, W = centroids_mw.shape
    if C != C_clusters:
        raise ValueError(f"C mismatch: {C} vs {C_clusters}")
    if M1 * W != M_features:
        raise ValueError(f"M_features mismatch: {M1 * W} vs {M_features}")

    if len(hidden_Ws) != len(hidden_lengths) + 1:
        raise ValueError(
            f"Expected {len(hidden_lengths)+1} weight matrices (hidden + output), got {len(hidden_Ws)}"
        )
    if len(hidden_bs) != len(hidden_lengths) + 1:
        raise ValueError(
            f"Expected {len(hidden_lengths)+1} bias vectors (hidden + output), got {len(hidden_bs)}"
        )

    in_dim = M_features
    for l_idx, (H_hidden, W_l, b_l) in enumerate(zip(H_hiddens, hidden_Ws[:-1], hidden_bs[:-1]), start=1):
        if W_l.shape != (in_dim, H_hidden):
            raise ValueError(f"W{l_idx} shape {W_l.shape} != ({in_dim}, {H_hidden})")
        if np.asarray(b_l).shape != (H_hidden,):
            raise ValueError(f"b{l_idx} shape {np.asarray(b_l).shape} != ({H_hidden},)")
        in_dim = H_hidden

    W_out = np.asarray(hidden_Ws[-1])
    b_out = np.asarray(hidden_bs[-1]).reshape(-1)
    if W_out.shape != (in_dim,):
        raise ValueError(f"W_out shape {W_out.shape} != ({in_dim},)")
    if b_out.shape != (1,):
        raise ValueError(f"b_out should be scalar-like, got shape {b_out.shape}")

    b_out_scalar = float(b_out[0])

    # Flatten centroids: (C, M1, W) -> (C, M)
    centroids_flat = centroids_mw.reshape(C, M1 * W)

    # --------------------
    # Bounds for ReLU big-M
    # --------------------
    if first_layer_trainable:
        LB_W1, UB_W1, LB_b1, UB_b1 = compute_weight_bounds_first_layer(
            W1=hidden_Ws[0],
            b1=hidden_bs[0],
            factor=w_factor,
        )
        LBs, UBs, _ = compute_relu_bounds_all_layers(
            centroids_flat=centroids_flat,
            hidden_Ws=hidden_Ws,
            hidden_bs=hidden_bs,
            beta=beta,
            eps=eps,
            LB_W1=LB_W1,
            UB_W1=UB_W1,
            LB_b1=LB_b1,
            UB_b1=UB_b1,
            use_interval_first_layer=True,
        )
    else:
        LBs, UBs, _ = compute_relu_bounds_all_layers(
            centroids_flat=centroids_flat,
            hidden_Ws=hidden_Ws,
            hidden_bs=hidden_bs,
            beta=beta,
            eps=eps,
        )

    # --------------------
    # Parameters
    # --------------------
    N0 = Parameter(cont, name="N0", description="Number of features allowed to remain active")
    N0.setRecords([int(n0_value)])

    xhat = Parameter(cont, name="xhat", domain=[c, m], description="Flattened centroids per cluster")
    xhat_records = [(ci + 1, mi + 1, float(centroids_flat[ci, mi]))
                    for ci in range(C_clusters) for mi in range(M_features)]
    xhat.setRecords(xhat_records)

    yhat = Parameter(cont, name="yhat", domain=c, description="Representative output per cluster")
    yhat.setRecords([(ci + 1, float(y_c[ci])) for ci in range(C_clusters)])

    alpha_p = Parameter(cont, name="alpha", domain=c, description="Cluster proportions")
    alpha_p.setRecords([(ci + 1, float(alpha[ci])) for ci in range(C_clusters)])

    W_params = []
    b_params = []
    LB_params = []
    UB_params = []

    n_hidden_layers = len(hs)

    # ---------- Hidden layers ----------
    for l_idx in range(n_hidden_layers):
        h_curr = hs[l_idx]
        H_curr = H_hiddens[l_idx]

        if l_idx == 0:
            dom_W = [m, h_curr]
            W_l = np.asarray(hidden_Ws[l_idx])
        else:
            h_prev = hs[l_idx - 1]
            dom_W = [h_prev, h_curr]
            W_l = np.asarray(hidden_Ws[l_idx])

        b_l = np.asarray(hidden_bs[l_idx]).reshape(-1)
        LB_l = np.asarray(LBs[l_idx]).reshape(-1)
        UB_l = np.asarray(UBs[l_idx]).reshape(-1)

        W_records = [(i_in + 1, i_out + 1, float(W_l[i_in, i_out]))
                     for i_in in range(W_l.shape[0]) for i_out in range(W_l.shape[1])]
        b_records = [(i_out + 1, float(b_l[i_out])) for i_out in range(H_curr)]

        if l_idx == 0 and first_layer_trainable:
            W_init = Parameter(cont, name="W1_init", domain=dom_W, description="Initial weights for hidden layer 1")
            W_init.setRecords(W_records)

            b_init = Parameter(cont, name="b1_init", domain=h_curr, description="Initial bias for hidden layer 1")
            b_init.setRecords(b_records)

            W_var = Variable(cont, name="W1_var", domain=dom_W, description="Trainable weights for hidden layer 1")

            W_var_rows = []
            for i_in in range(W_l.shape[0]):
                for i_out in range(W_l.shape[1]):
                    W_var_rows.append({
                        "m": i_in + 1,
                        "h1": i_out + 1,
                        "level": float(W_l[i_in, i_out]),
                        "lower": float(LB_W1[i_in, i_out]),
                        "upper": float(UB_W1[i_in, i_out]),
                    })
            W_var.setRecords(pd.DataFrame(W_var_rows))

            b_var = Variable(cont, name="b1_var", domain=h_curr, description="Trainable bias for hidden layer 1")
            b_var_rows = [{
                "h1": i_out + 1,
                "level": float(b_l[i_out]),
                "lower": float(LB_b1[i_out]),
                "upper": float(UB_b1[i_out]),
            } for i_out in range(H_curr)]
            b_var.setRecords(pd.DataFrame(b_var_rows))

            W_params.append(W_var)
            b_params.append(b_var)
        else:
            W_param = Parameter(cont, name=f"W{l_idx+1}", domain=dom_W, description=f"Weights for hidden layer {l_idx+1}")
            W_param.setRecords(W_records)
            W_params.append(W_param)

            b_param = Parameter(cont, name=f"b{l_idx+1}", domain=h_curr, description=f"Bias for hidden layer {l_idx+1}")
            b_param.setRecords(b_records)
            b_params.append(b_param)

        LB_param = Parameter(cont, name=f"LB{l_idx+1}", domain=h_curr, description=f"Lower bound for z in hidden layer {l_idx+1}")
        LB_param.setRecords([(i_out + 1, float(LB_l[i_out])) for i_out in range(H_curr)])
        LB_params.append(LB_param)

        UB_param = Parameter(cont, name=f"UB{l_idx+1}", domain=h_curr, description=f"Upper bound for z in hidden layer {l_idx+1}")
        UB_param.setRecords([(i_out + 1, float(UB_l[i_out])) for i_out in range(H_curr)])
        UB_params.append(UB_param)

    # ---------- Output layer ----------
    h_last = hs[-1]
    H_last = H_hiddens[-1]

    W_out_param = Parameter(cont, name="W_out", domain=h_last, description="Weights to scalar output")
    W_out_param.setRecords([(i + 1, float(W_out[i])) for i in range(H_last)])

    b_out_param = Parameter(cont, name="b_out", description="Output bias (scalar)")
    b_out_param.setRecords([b_out_scalar])

    # --------------------
    # Variables
    # --------------------
    if mode == TRUST_MODE_SENSORS:
        S0 = Variable(cont, name="S0", domain=s, type="binary", description="Sensor selection")
    elif mode == TRUST_MODE_WINDOWS:
        W0 = Variable(cont, name="W0", domain=w, type="binary", description="Window selection")

    Z0 = Variable(cont, name="Z0", domain=m, type="binary", description="Feature selection indicator")

    z_vars = []
    a_vars = []
    sigma_vars = []

    for l_idx, h_curr in enumerate(hs, start=1):
        z_l = Variable(cont, name=f"z{l_idx}", domain=[c, h_curr], description=f"Pre-activation for hidden layer {l_idx}")
        a_l = Variable(cont, name=f"a{l_idx}", domain=[c, h_curr], type="positive", description=f"Activation (ReLU) for hidden layer {l_idx}")
        sigma_l = Variable(cont, name=f"sigma{l_idx}", domain=[c, h_curr], type="binary", description=f"ReLU region indicator for hidden layer {l_idx}")
        z_vars.append(z_l)
        a_vars.append(a_l)
        sigma_vars.append(sigma_l)

    y_pred = Variable(cont, name="y_pred", domain=c, description="Predicted output for each centroid")
    D = Variable(cont, name="D", domain=c, type="positive", description="Absolute error per centroid")

    # --------------------
    # Equations
    # --------------------
    if mode == TRUST_MODE_SENSORS:
        Z0_S0_links = []
        for m_idx in range(M_features):
            s_idx = (m_idx // W) + 1
            eq = Equation(cont, name=f"link_Z0_S0_{m_idx+1}", description=f"Feature {m_idx+1} tied to sensor {s_idx}")
            eq[...] = Z0[m_idx + 1] == S0[s_idx]
            Z0_S0_links.append(eq)

    elif mode == TRUST_MODE_WINDOWS:
        Z0_W0_links = []
        for m_idx in range(M_features):
            w_idx = (m_idx % W) + 1
            eq = Equation(cont, name=f"link_Z0_W0_{m_idx+1}", description=f"Feature {m_idx+1} tied to window {w_idx}")
            eq[...] = Z0[m_idx + 1] == W0[w_idx]
            Z0_W0_links.append(eq)

    # Layer 1 definition
    if not first_layer_trainable:
        z1_def = Equation(cont, name="z1_def", domain=[c, hs[0]], description="Layer 1 pre-activation (fixed W1, b1)")
        z1_def[c, hs[0]] = z_vars[0][c, hs[0]] == Sum(m, W_params[0][m, hs[0]] * xhat[c, m] * Z0[m]) + b_params[0][hs[0]]
    else:
        W1Z = Variable(cont, name="W1Z", domain=[m, hs[0]], description="Aux var for W1_var * Z0 linearization")

        lin1 = Equation(cont, name="lin_W1Z_ubZ", domain=[m, hs[0]], description="W1Z <= UB_W1 * Z0")
        lin2 = Equation(cont, name="lin_W1Z_lbZ", domain=[m, hs[0]], description="W1Z >= LB_W1 * Z0")
        lin3 = Equation(cont, name="lin_W1Z_ubW", domain=[m, hs[0]], description="W1Z <= W1_var - LB_W1*(1-Z0)")
        lin4 = Equation(cont, name="lin_W1Z_lbW", domain=[m, hs[0]], description="W1Z >= W1_var - UB_W1*(1-Z0)")

        LB_W1_p = Parameter(cont, name="LB_W1", domain=[m, hs[0]], description="Lower bounds for W1_var")
        UB_W1_p = Parameter(cont, name="UB_W1", domain=[m, hs[0]], description="Upper bounds for W1_var")

        LB_W1_p.setRecords([(i_in + 1, i_out + 1, float(LB_W1[i_in, i_out]))
                            for i_in in range(M_features) for i_out in range(H_hiddens[0])])
        UB_W1_p.setRecords([(i_in + 1, i_out + 1, float(UB_W1[i_in, i_out]))
                            for i_in in range(M_features) for i_out in range(H_hiddens[0])])

        lin1[m, hs[0]] = W1Z[m, hs[0]] <= UB_W1_p[m, hs[0]] * Z0[m]
        lin2[m, hs[0]] = W1Z[m, hs[0]] >= LB_W1_p[m, hs[0]] * Z0[m]
        lin3[m, hs[0]] = W1Z[m, hs[0]] <= W_params[0][m, hs[0]] - LB_W1_p[m, hs[0]] * (1 - Z0[m])
        lin4[m, hs[0]] = W1Z[m, hs[0]] >= W_params[0][m, hs[0]] - UB_W1_p[m, hs[0]] * (1 - Z0[m])

        z1_def = Equation(cont, name="z1_def", domain=[c, hs[0]], description="Layer 1 pre-activation (trainable W1)")
        z1_def[c, hs[0]] = z_vars[0][c, hs[0]] == Sum(m, W1Z[m, hs[0]] * xhat[c, m]) + b_params[0][hs[0]]

    # Layers 2..L
    for l_idx in range(1, n_hidden_layers):
        h_prev = hs[l_idx - 1]
        h_curr = hs[l_idx]
        eq = Equation(cont, name=f"z{l_idx+1}_def", domain=[c, h_curr], description=f"Layer {l_idx+1} pre-activation")
        eq[c, h_curr] = z_vars[l_idx][c, h_curr] == Sum(h_prev, W_params[l_idx][h_prev, h_curr] * a_vars[l_idx - 1][c, h_prev]) + b_params[l_idx][h_curr]

    # ReLU big-M constraints
    for l_idx, h_curr in enumerate(hs, start=1):
        z_l = z_vars[l_idx - 1]
        a_l = a_vars[l_idx - 1]
        sigma_l = sigma_vars[l_idx - 1]
        LB_l = LB_params[l_idx - 1]
        UB_l = UB_params[l_idx - 1]

        relu_lin = Equation(cont, name=f"relu_linear_{l_idx}", domain=[c, h_curr], description=f"ReLU linear part (layer {l_idx})")
        relu_lin[c, h_curr] = a_l[c, h_curr] >= z_l[c, h_curr]

        relu_lower = Equation(cont, name=f"relu_bigM_lower_{l_idx}", domain=[c, h_curr], description=f"ReLU big-M lower (layer {l_idx})")
        relu_lower[c, h_curr] = a_l[c, h_curr] <= z_l[c, h_curr] - LB_l[h_curr] * (1 - sigma_l[c, h_curr])

        relu_upper = Equation(cont, name=f"relu_bigM_upper_{l_idx}", domain=[c, h_curr], description=f"ReLU big-M upper (layer {l_idx})")
        relu_upper[c, h_curr] = a_l[c, h_curr] <= UB_l[h_curr] * sigma_l[c, h_curr]

    # Output layer
    a_last = a_vars[-1]
    y_def = Equation(cont, name="y_def", domain=c, description="Output prediction")
    y_def[c] = y_pred[c] == Sum(h_last, W_out_param[h_last] * a_last[c, h_last]) + b_out_param

    # Absolute error
    err_pos = Equation(cont, name="err_pos", domain=c, description="D >= y_pred - yhat")
    err_pos[c] = D[c] >= y_pred[c] - yhat[c]

    err_neg = Equation(cont, name="err_neg", domain=c, description="D >= yhat - y_pred")
    err_neg[c] = D[c] >= yhat[c] - y_pred[c]

    # Selection constraint
    feat_sel = Equation(cont, name="feat_sel", description="Limit number of selected inputs")
    if mode == TRUST_MODE_FULL:
        if n0_value > M_features:
            raise ValueError(f"N0={n0_value} > M_features={M_features}")
        feat_sel[...] = Sum(m, Z0[m]) == N0
    elif mode == TRUST_MODE_SENSORS:
        if n0_value > S_sensors:
            raise ValueError(f"N0={n0_value} > S_sensors={S_sensors}")
        feat_sel[...] = Sum(s, S0[s]) == N0
    elif mode == TRUST_MODE_WINDOWS:
        if n0_value > W_windows:
            raise ValueError(f"N0={n0_value} > W_windows={W_windows}")
        feat_sel[...] = Sum(w, W0[w]) == N0

    # Objective
    obj_expr = Sum(c, alpha_p[c] * D[c])

    # ------------------------------------------------------------
    # Baseline control: forbid being better than baseline surrogate
    # ------------------------------------------------------------
    if baseline_mode == BASELINE_MODE_CAPPED:
        mae_floor = Parameter(
            cont,
            name="mae_floor",
            description="Lower bound on surrogate MAE (baseline cap)"
        )
        mae_floor.setRecords([
            float(baseline_mae) * (1.0 - float(baseline_slack))
        ])

        baseline_floor_eq = Equation(
            cont,
            name="baseline_mae_floor",
            description="Prevent surrogate MAE from improving over baseline"
        )
        baseline_floor_eq[...] = obj_expr >= mae_floor


    trust_model = Model(
        cont,
        name="trust_model",
        equations=cont.getEquations(),
        problem=Problem.MIP,
        sense=Sense.MIN,
        objective=obj_expr,
    )

    return {
        "container": cont,
        "model": trust_model,

        # Primary selection vars
        "Z0": Z0,
        "S0": S0 if mode == TRUST_MODE_SENSORS else None,
        "W0": W0 if mode == TRUST_MODE_WINDOWS else None,

        # Configuration echo
        "mode": mode,
        "milp_mode": milp_mode,
        "first_layer_trainable": first_layer_trainable,
        "baseline_mode": baseline_mode,
        "baseline_mae": float(baseline_mae) if baseline_mae is not None else None,
        "baseline_slack": float(baseline_slack),

        # Useful parameters / dimensions
        "dims": {
            "C": int(C_clusters),
            "M_features": int(M_features),
            "M_sensors": int(centroids_mw.shape[1]),
            "W_windows": int(centroids_mw.shape[2]),
            "hidden_lengths": list(hidden_lengths),
        },

        # Baseline constraint handle (optional)
        "baseline_floor_eq": baseline_floor_eq if baseline_mode == BASELINE_MODE_CAPPED else None,

        # Internal vars that the pipeline often needs
        "other_vars": {
            # Forward vars
            "z": z_vars,
            "a": a_vars,
            "sigma": sigma_vars,
            "y_pred": y_pred,
            "D": D,

            # Weights/biases used in equations (Parameter or Variable in MILP-2 layer 1)
            "W_params": W_params,
            "b_params": b_params,

            # Direct access to first-layer objects
            "W1": W_params[0],  # Parameter in MILP-1, Variable in MILP-2
            "b1": b_params[0],
            "W1_var": W_params[0] if first_layer_trainable else None,
            "b1_var": b_params[0] if first_layer_trainable else None,
        },
    }
