from __future__ import annotations

from pathlib import Path

import numpy as np

from public_time_trust.pipeline import run_trust_with_baseline_training
from public_time_trust.utils.milp_utils import TRUST_MODE_SENSORS, MLP_REBUILD


def main() -> None:
    rng = np.random.default_rng(2026)

    # Synthetic SxW data: N x S x W
    n_train, n_val = 160, 48
    sensors, windows = 6, 10

    X_train = rng.normal(size=(n_train, sensors, windows)).astype(np.float32)
    X_val = rng.normal(size=(n_val, sensors, windows)).astype(np.float32)

    # A simple target correlated with a few sensor-window locations.
    y_train = (
        0.7 * X_train[:, 0, 0]
        - 0.4 * X_train[:, 2, 4]
        + 0.6 * X_train[:, 4, 7]
        + 0.05 * rng.normal(size=(n_train,))
    ).astype(np.float32)
    y_val = (
        0.7 * X_val[:, 0, 0]
        - 0.4 * X_val[:, 2, 4]
        + 0.6 * X_val[:, 4, 7]
        + 0.05 * rng.normal(size=(n_val,))
    ).astype(np.float32)

    result = run_trust_with_baseline_training(
        dataset_name="SYNTH",
        window_tag="w10",
        results_root=Path("results"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hidden_layers=(12, 8),
        mode=TRUST_MODE_SENSORS,
        C=16,
        beta=0.4,
        mlp_mode=MLP_REBUILD,
        milp_time_cap=30,
        learning_rate=1e-3,
        epochs=5,
        batch_size=32,
        verbose=True,
        resume=False,
    )

    print("TRUST run completed")
    print(f"Output dir: {result['out_dir']}")


if __name__ == "__main__":
    main()
