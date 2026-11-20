"""HyperCAST: hypercomplex neural layers, executable model graphs, and forecasting evaluation.

HyperCAST provides explicit structure-constant hypercomplex layers in PyTorch,
a composable v1 pipeline and v2 FX-lowered graph architecture compiler, and
leakage-safe chronological time-series evaluation workflows.

Public Subsystems
-----------------
- ``hypercast4d.algebras`` — Finite-dimensional real algebras (2D, 4D, 8D)
  and bilinear product kernels.
- ``hypercast4d.layers`` — [`HyperDense`][hypercast4d.layers.HyperDense] and
  [`ShapePreservingHyperDense`][hypercast4d.layers.ShapePreservingHyperDense]
  PyTorch modules.
- ``hypercast4d.architecture`` — Schema v1 sequential architecture compiler,
  layer catalog, and presets.
- ``hypercast4d.graph_architecture`` — Schema v2 executable DAG compiler,
  source-symbolic lowering conversion, and node inspection.
- ``hypercast4d.model_editing`` — Programmatic layer insertion, Dense ↔
  HyperDense width-preserving swaps, and graph wiring.
- ``hypercast4d.data`` — Chronological split construction and training-only
  min–max normalization.
- ``hypercast4d.training`` — Deterministic PyTorch training, inference
  benchmarking, and error metrics.
- ``hypercast4d.diagnostics`` — Per-lead directional, correlation, tail-error,
  and large-move forecast diagnostics.
- ``hypercast4d.playground`` / ``hypercast4d.cli`` — Local FastAPI workbench
  server and ``hypercast`` command-line interface.

Internal Benchmark Modules
--------------------------
Modules named ``tslib15``, ``hyperdense_transfer*``, ``remaining_*``,
``representation_pilot``, ``native_transfer``, and ``crossformer_baseline``
implement frozen experimental protocols for archived benchmark studies rather
than general-purpose library APIs.

Examples
--------
>>> import torch
>>> from hypercast4d import HyperDense, get_algebra
>>> layer = HyperDense(in_features=4, out_features=2, algebra="quaternion")
>>> x = torch.randn(8, 16, 16)  # 4 components * 4 hypercomplex input units
>>> y = layer(x)
>>> tuple(y.shape)
(8, 16, 8)
"""

from .algebras import (
    ALGEBRAS,
    COMPONENT_COUNT,
    PAPER_COMPONENT_COUNT,
    Algebra,
    get_algebra,
)
from .architecture import (
    ArchitectureError,
    architecture_hash,
    build_architecture,
    layer_catalog,
    normalize_architecture_spec,
    presets,
    validate_architecture,
)
from .data import (
    MinMaxStats,
    PreparedData,
    WindowSplit,
    load_dataset,
    load_paper_data,
    prepare_windows,
)
from .datasets import (
    DATASETS,
    DatasetSpec,
    dataset_catalog,
    download_dataset,
    get_dataset_spec,
)
from .graph_architecture import (
    GraphForecaster,
    convert_to_graph,
    describe_graph,
)
from .layers import HyperDense, ShapePreservingHyperDense
from .multilayer_hyperdense import (
    BlockDiagonalLinear,
    HyperChannelBlock,
    HyperChannelMixer,
    HyperGate,
    HyperRMSNorm,
    LowRankLinear,
    build_multilayer_model,
)
from .training import (
    FitResult,
    error_metrics,
    fit_model,
    predict,
    seed_everything,
)

__all__ = [
    "ALGEBRAS",
    "Algebra",
    "ArchitectureError",
    "BlockDiagonalLinear",
    "COMPONENT_COUNT",
    "DATASETS",
    "DatasetSpec",
    "FitResult",
    "GraphForecaster",
    "HyperChannelBlock",
    "HyperChannelMixer",
    "HyperDense",
    "HyperGate",
    "HyperRMSNorm",
    "LowRankLinear",
    "MinMaxStats",
    "PAPER_COMPONENT_COUNT",
    "PreparedData",
    "ShapePreservingHyperDense",
    "WindowSplit",
    "architecture_hash",
    "build_architecture",
    "build_multilayer_model",
    "convert_to_graph",
    "dataset_catalog",
    "describe_graph",
    "download_dataset",
    "error_metrics",
    "fit_model",
    "get_algebra",
    "get_dataset_spec",
    "layer_catalog",
    "load_dataset",
    "load_paper_data",
    "normalize_architecture_spec",
    "prepare_windows",
    "predict",
    "presets",
    "seed_everything",
    "validate_architecture",
]
__version__ = "0.2.0"
