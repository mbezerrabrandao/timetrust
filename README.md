# public_time_trust

Minimal, publishable subset of the TRUST pipeline focused on:
- iterative TRUST selection (full, sensors, windows),
- MILP-1 and MILP-2 formulation via GAMSPy,
- surrogate MLP build/training,
- run artifacts/checkpoints.

## What is included
- `public_time_trust/pipeline.py`: end-to-end TRUST loop.
- `public_time_trust/milp/trust_milp.py`: MILP model builder.
- `public_time_trust/utils/mlp_utils.py`: MLP model creation helpers.
- `public_time_trust/utils/trust_utils.py`: centroiding, training, importance, warm-start.
- `public_time_trust/utils/milp_utils.py`: constants and MILP bound helpers.
- `public_time_trust/pipeline_artifacts.py`: summarize run outputs.
- `examples/run_synthetic.py`: synthetic SxW demo.

## Install
```bash
pip install -e .
```

## Solver requirements
This package uses `gamspy` and expects a MILP solver backend (e.g. Gurobi through your GAMS/GAMSPy setup).

If solver dependencies are missing, code import/training can still work, but MILP solve calls will fail at runtime.

## Minimal usage
```python
import numpy as np
from public_time_trust.pipeline import run_trust_with_baseline_training
from public_time_trust.utils.milp_utils import TRUST_MODE_SENSORS

rng = np.random.default_rng(2026)
X_train = rng.normal(size=(256, 8, 12)).astype(np.float32)
X_val = rng.normal(size=(64, 8, 12)).astype(np.float32)
y_train = rng.normal(size=(256,)).astype(np.float32)
y_val = rng.normal(size=(64,)).astype(np.float32)

result = run_trust_with_baseline_training(
    dataset_name="SYNTH",
    window_tag="w12",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    hidden_layers=(16, 8),
    mode=TRUST_MODE_SENSORS,
    C=20,
    epochs=5,
    batch_size=64,
)

print(result["out_dir"])
```

## Run the example
```bash
python examples/run_synthetic.py
```
