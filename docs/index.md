# HyperCAST Documentation

**HyperCAST** (`hypercast4d`) is a PyTorch library, command-line toolkit, and interactive visual workbench for constructing, inspecting, and evaluating **hypercomplex neural networks** in multivariate time-series forecasting.

It combines:

1. **Drop-in PyTorch hypercomplex layers** ([`HyperDense`][hypercast4d.layers.HyperDense] and [`ShapePreservingHyperDense`][hypercast4d.layers.ShapePreservingHyperDense]) backed by explicit structure-constant tensors across six finite-dimensional real algebras (2D, 4D, and 8D).
2. **Executable model graphs** ([`hypercast4d.architecture`][hypercast4d.architecture] and [`hypercast4d.graph_architecture`][hypercast4d.graph_architecture]) supporting both sequential pipelines (Schema v1) and symbolically lowered PyTorch FX directed acyclic graphs (Schema v2), including 15 composable TSLib model cores.
3. **Leakage-safe forecasting evaluation** ([`hypercast4d.data`][hypercast4d.data], [`hypercast4d.training`][hypercast4d.training], and [`hypercast4d.diagnostics`][hypercast4d.diagnostics]) enforcing chronological target splits, training-only normalization, persistence baselines, per-lead diagnostics, and one-shot held-out final testing.
4. **Unified visual & CLI workflows** ([`hypercast4d-playground`][hypercast4d.playground] and [`hypercast`][hypercast4d.cli]) sharing identical YAML/JSON specifications, shape validation, weight-matrix inspectors, and local/Modal/GCP execution backends.

---

## Installation

Python 3.10 or newer is required (Python 3.12 is tested).

```bash
cd hypercast4d
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,docs]'
```

To build or preview this documentation site locally:

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Documentation Map

| Section | What You Will Find |
|---|---|
| **[Quickstart & Tutorials](quickstart.md)** | Copy-pasteable Python examples for `HyperDense`, `ShapePreservingHyperDense`, Schema v1/v2 graph editing, leakage-safe training, CLI usage, and the visual Playground. |
| **[Algorithms & Mathematics](algorithms.md)** | Structure constants and multiplication tables for all 6 algebras, real block-matrix equivalence, parameter complexity ($dnm$ vs. $d^2nm$), symbolic FX lowering, chronological split rules, and diagnostic metric definitions. |
| **[Supported Components & Models](components.md)** | Parameter bounds, defaults, and tensor-shape contracts for all pipeline blocks, graph operators, input representations, forecast heads, and the 25 built-in presets. |
| **[Schemas, YAML & REST API](configuration.md)** | Complete reference for `configs/evaluation.yaml`, `configs/paper_reproduction.yaml`, Schema v1 & v2 YAML/JSON formats, environment variables, and loopback HTTP endpoints (`/api/v1/*`). |
| **[Command-Line Interface](cli.md)** | Guide to `hypercast` (`model`, `layer`, `graph`, `experiment`, `job`, `server`), exit codes, and machine-readable `--json` output. |
| **[Extending HyperCAST](extending.md)** | Step-by-step developer guide for adding custom algebras, custom graph layers/operators, and new forecasting backbones. |
| **[API Reference](api/index.md)** | Module-by-module class and function reference generated directly from source docstrings and type annotations. |

---

## Package Architecture at a Glance

```text
hypercast4d
├── algebras.py            # 6 real algebras (2D/4D/8D) & bilinear product kernels
├── layers.py              # HyperDense & ShapePreservingHyperDense (nn.Module)
├── visualizations.py      # Exact component & effective block-matrix snapshots
├── architecture.py        # Schema v1 sequential compiler, catalog & 25 presets
├── graph_lowering.py      # Symbolic PyTorch FX tracer & TSLib lowering kernels
├── graph_architecture.py  # Schema v2 executable DAG compiler & port inspector
├── model_editing.py       # Atomic file I/O, cycle-safe wiring & Dense ↔ HyperDense swaps
├── models.py              # Canonical 4D paper architectures (Figure 3) & Linear baseline
├── upstream_models.py     # Pinned TSLib-15 sequence-to-sequence adapters
├── research_models.py     # Lightweight DLinear / PatchTST / iTransformer blocks
├── data.py                # Chronological windowing & training-only MinMaxStats
├── training.py            # Seeded Adam loop, early stopping & inference benchmarks
├── diagnostics.py         # Directional, correlation, p95 & large-move diagnostics
├── experiment.py          # Practical chronological-v1 matrix CLI (hypercast4d-run)
├── reproduction.py        # Archived notebook GridSearchCV CLI (hypercast4d-reproduce)
├── playground.py          # FastAPI server, JobManager queue & OpenAPI (/docs)
└── cli.py                 # Unified `hypercast` model/graph/experiment CLI
```
