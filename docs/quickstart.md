# Quickstart & Tutorials

HyperCAST can be used at three levels depending on your workflow:

1. **As a standalone PyTorch layer library** (`HyperDense`, `ShapePreservingHyperDense`, `get_algebra`) inside your own models.
2. **As a Python architecture & evaluation API** (`build_architecture`, `convert_to_graph`, `edit_layer`, `prepare_windows`, `fit_model`).
3. **Through the visual Playground or `hypercast` CLI** for interactive architecture surgery and reproducible experiment management.

---

## 1. Using `HyperDense` in Any PyTorch Model

[`HyperDense`][hypercast4d.layers.HyperDense] is a standard `torch.nn.Module`. For an algebra of dimension $d$ (for example $d = 4$ for `"quaternion"`), an input of last-axis width $d \times n$ is mapped to $d \times m$ using $d \cdot n \cdot m$ kernel weights instead of $d^2 \cdot n \cdot m$:

```python
import torch
from hypercast4d import HyperDense, ShapePreservingHyperDense, get_algebra

# 1. Inspect an algebra directly
quat = get_algebra("quaternion")
print(quat.component_count)  # 4

# 2. Strict hypercomplex projection: 4 * 8 = 32 in -> 4 * 16 = 64 out
layer = HyperDense(in_features=8, out_features=16, algebra="quaternion", bias=True)
x = torch.randn(32, 24, 32)  # [batch, time, features]
y = layer(x)
print(y.shape)  # torch.Size([32, 24, 64])

# Parameter comparison against nn.Linear(32, 64):
# HyperDense: 4 * 8 * 16 + 64 = 576 parameters
# nn.Linear:  32 * 64 + 64    = 2,112 parameters
```

### Handling Arbitrary Channel Counts with `ShapePreservingHyperDense`

Real-world multivariate datasets rarely have feature counts divisible by 4 or 8 (for example, ETTh1 has 7 channels). [`ShapePreservingHyperDense`][hypercast4d.layers.ShapePreservingHyperDense] right-pads the last axis with zeros to $\lceil C / d \rceil \cdot d$, applies a square `HyperDense` layer, and crops back to $C$:

```python
adapter = ShapePreservingHyperDense(width=7, algebra="octonion")
print(adapter.shape_fit)
# {'axis': -1, 'input_width': 7, 'padded_width': 8, 'units': 1,
#  'output_width': 7, 'padding': 1, 'crop': 1}

etth1_batch = torch.randn(16, 32, 7)
encoded = adapter(etth1_batch)
print(encoded.shape)  # torch.Size([16, 32, 7])
```

---

## 2. Building, Lowering & Editing Architectures in Python

HyperCAST provides 25 ready-to-use presets (5 locked paper models, 5 research/residual templates, and 15 TSLib-core adaptations). Any Schema v1 pipeline can be lowered into an explicit Schema v2 operator graph (`GraphSpec`) and edited programmatically:

```python
import copy
import torch
from hypercast4d import (
    build_architecture,
    convert_to_graph,
    presets,
    validate_architecture,
)
from hypercast4d.model_editing import edit_layer
from hypercast4d.visualizations import model_weights

# Load an editable TSLib TSMixer preset
tsmixer = copy.deepcopy(
    next(p for p in presets() if p["preset_id"] == "tslib-tsmixer")
)

# Convert Schema v1 pipeline -> Schema v2 executable FX graph
graph = convert_to_graph(tsmixer, window=10, horizon=1)
info = validate_architecture(graph, window=10, horizon=1)
print(f"Nodes: {len(graph['nodes'])}, Parameters: {info['parameters']:,}")

# Swap an internal Linear node to a width-matched Quaternion HyperDense layer
cells = [{"window": 10, "horizon": 1}, {"window": 20, "horizon": 5}]
edited = edit_layer(
    graph,
    "replace",
    node_id="layers_0_model_model_0_temporal_0",
    kind="hyper_dense",
    params={"algebra": "quaternion"},
    cells=cells,
)

# Compile into an executable torch.nn.Module and run a forward pass
model = build_architecture(edited, window=20, horizon=5, features=4)
forecast = model(torch.randn(4, 20, 4))
print(forecast.shape)  # torch.Size([4, 5])

# Inspect component matrices and the structured real block matrix
snapshot = model_weights(model)["layers"][0]
print(snapshot["algebra"], snapshot["weight_parameters"], snapshot["dense_parameters"])
```

---

## 3. Running a Leakage-Safe Chronological Evaluation in Python

[`prepare_windows`][hypercast4d.data.prepare_windows] guarantees that min–max scaling statistics are fitted strictly on training rows and that no target horizon crosses a split boundary:

```python
import numpy as np
import pandas as pd
import torch
from hypercast4d import (
    HyperDense,
    error_metrics,
    fit_model,
    predict,
    prepare_windows,
)
from torch import nn

# Create a synthetic 4-channel series (or call load_paper_data("data/raw/paper_data.xlsx"))
t = np.linspace(0, 40, 600)
frame = pd.DataFrame(
    {
        "Target": np.sin(t) + 0.05 * t,
        "SeriesB": np.cos(t),
        "SeriesC": np.sin(0.5 * t),
        "SeriesD": np.cos(0.5 * t),
    },
    index=pd.date_range("2022-01-01", periods=600, freq="D"),
)

prepared = prepare_windows(
    frame, window=16, horizon=4, train_fraction=0.70, validation_fraction=0.15
)

model = nn.Sequential(
    HyperDense(in_features=1, out_features=8, algebra="quaternion"),
    nn.ReLU(),
    nn.Flatten(start_dim=1),
    nn.Linear(16 * 32, 4),
)

result = fit_model(
    model,
    prepared.train.as_dataset(),
    prepared.validation.as_dataset(),
    seed=7,
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-7,
    adam_amsgrad=False,
    loss_name="mse",
    shuffle=True,
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
    restore_best_weights=True,
    device=torch.device("cpu"),
)

# Inverse-transform scaled validation predictions back to original Target units
pred_scaled = predict(model, prepared.validation.as_dataset(), torch.device("cpu"), batch_size=64)
pred = prepared.scaler.inverse_target(pred_scaled)
actual = prepared.scaler.inverse_target(prepared.validation.y)
print(result.best_epoch, error_metrics(pred, actual))
```

---

## 4. Visual Playground & CLI Workflows

=== "Interactive Playground"

    ```bash
    # Download the 4-series Copper dataset once
    hypercast4d-download

    # Launch the local FastAPI + React Flow workbench on http://127.0.0.1:8765
    hypercast4d-playground
    ```

    - Interactive Swagger/OpenAPI docs are simultaneously served at `http://127.0.0.1:8765/docs`.
    - Expand any model stage, click **+** on a connection to insert an auto-fitted `HyperDense` block, open **Inspect weight structure**, and submit validation runs locally, on Modal, or on GCP GPUs.

=== "`hypercast` CLI"

    ```bash
    # Create an editable model, expand to Schema v2 graph, swap a layer, and dry-run
    hypercast model create tslib-tsmixer --out tsmixer.yaml
    hypercast model expand tsmixer.yaml --out tsmixer-graph.yaml --cells 10/1,20/5
    hypercast layer replace tsmixer-graph.yaml layers_0_model_model_0_temporal_0 \
      --type hyper_dense --set algebra=quaternion --cells 10/1,20/5
    hypercast experiment run tsmixer-graph.yaml --preset quick --dry-run
    ```

=== "Paper Evaluation Runners"

    ```bash
    # Bounded leakage-safe evaluation (chronological-v1)
    hypercast4d-run --quick

    # Archived supplementary notebook GridSearchCV reproduction
    hypercast4d-reproduce --quick
    ```
