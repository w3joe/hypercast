# API Reference

The `hypercast4d` Python package exposes hypercomplex algebras, PyTorch layers, architecture/graph compilers, data preparation, deterministic training, forecast diagnostics, and local/cloud experiment services.

## Top-Level Package (`hypercast4d`)

Commonly used classes and functions are re-exported directly at the package root (`from hypercast4d import HyperDense, ShapePreservingHyperDense, build_architecture, ...`).

::: hypercast4d
    options:
      members: false

---

## Submodule Index

| Page | Modules Covered |
|---|---|
| **[Algebras, Layers & Weights](algebras_and_layers.md)** | `hypercast4d.algebras`, `hypercast4d.layers`, `hypercast4d.visualizations` |
| **[Architectures, Lowering & Editing](architectures_and_graphs.md)** | `hypercast4d.architecture`, `hypercast4d.graph_architecture`, `hypercast4d.graph_lowering`, `hypercast4d.model_editing`, `hypercast4d.internal_editing`, `hypercast4d.internal_graph` |
| **[Forecasting Models & Catalog](models.md)** | `hypercast4d.models`, `hypercast4d.upstream_models`, `hypercast4d.research_models`, `hypercast4d.method_collection` |
| **[Data, Training & Diagnostics](data_and_training.md)** | `hypercast4d.data`, `hypercast4d.training`, `hypercast4d.diagnostics`, `hypercast4d.download`, `hypercast4d.experiment`, `hypercast4d.experiment_controls`, `hypercast4d.reproduction` |
| **[Playground, CLI & Cloud Compute](interfaces_and_compute.md)** | `hypercast4d.playground`, `hypercast4d.playground_runner`, `hypercast4d.cli`, `hypercast4d.workspace_runtime`, `hypercast4d.compute`, `hypercast4d.gcp_compute`, `hypercast4d.comparison_archives`, `hypercast4d.dashboard` |
