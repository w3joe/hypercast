# Extending HyperCAST

HyperCAST is structured around explicit data tables and registries rather than dynamic code execution, making it straightforward to extend with new algebras, custom layers/graph operators, or forecasting backbones.

---

## 1. Adding a New Real Algebra

Every algebra in [`hypercast4d.algebras`][hypercast4d.algebras] is an immutable [`Algebra(name, constants)`][hypercast4d.algebras.Algebra] instance whose `constants` tensor has shape `(d, d, d)` with $e_0 = 1$ as the identity.

### Step-by-Step

1. **Define the structure constants in `src/hypercast4d/algebras.py`** using `_constants(dimension, products)`, where `products[(a, b)] = (sign, c)` encodes $e_a e_b = \text{sign} \cdot e_c$ for $a, b \ge 1$ ($e_0$ identity entries are populated automatically):

    ```python
    # Example: Dual numbers R[eps] / (eps^2 = 0), dimension d = 2
    _DUAL = _constants(2, {(1, 1): (0.0, 0)})
    ```

2. **Register the algebra in `ALGEBRAS`** (and optional `_ALIASES` in `src/hypercast4d/algebras.py`, plus `DEFAULT_ALGEBRA_DIMENSIONS` in `web/src/layerSwap.ts` if exposing Dense ↔ HyperDense UI swaps):

    ```python
    ALGEBRAS["dual"] = Algebra("dual", _DUAL)
    ```

3. **Automatic propagation**:
   - [`HyperDense`][hypercast4d.layers.HyperDense] and [`ShapePreservingHyperDense`][hypercast4d.layers.ShapePreservingHyperDense] immediately accept `algebra="dual"`.
   - `architecture.ALGEBRAS`, `layer_catalog()`, `/api/v1/catalog`, `hypercast layer types`, and `visualizations.weight_snapshot()` automatically pick up the new algebra and its dimension.

4. **Add unit tests in `tests/test_algebras.py` and `tests/test_layers.py`** verifying basis products, gradient flow against [`explicit_real_matrix`][hypercast4d.experiment_controls.explicit_real_matrix], and shape preservation.

---

## 2. Adding a New Layer or Graph Operator

To expose a custom PyTorch `nn.Module` in both Schema v1 pipelines and Schema v2 graphs:

1. **Implement the `nn.Module`** (for example in `src/hypercast4d/layers.py` or `src/hypercast4d/architecture.py`) operating on channel-last `[B, T, C]` sequences or `[B, C]` vectors.
2. **Validate parameters in `src/hypercast4d/architecture.py`**:
   - Add the type key (e.g. `"gated_mlp"`) to `LAYER_TYPES`.
   - Add parameter normalization and bounds in `_normalize_layer_params()`.
   - Instantiate the module and return `(module, next_shape, receptive_addition)` in `_build_layer()`.
   - Add a UI catalog entry in `layer_catalog()` so the Playground **Add** palette and `hypercast layer types` list its label and defaults.
3. **Enable Schema v2 graph execution**:
   - Add `"gated_mlp"` to `NEW_OPS` in `src/hypercast4d/graph_architecture.py` (`GraphForecaster._new()` automatically delegates unary `NEW_OPS` to `_normalize_layer_params` and `_build_layer`!).
   - If the module should remain an atomic leaf when lowering a Schema v1 pipeline to a Schema v2 graph, add its class name to `SourceTracer.is_leaf_module()` in `src/hypercast4d/graph_lowering.py`.

---

## 3. Wrapping a New Forecasting Backbone

HyperCAST supports two patterns for multi-layer forecasting models:

- **Lightweight native blocks (`src/hypercast4d/research_models.py`)**: Self-contained `nn.Module` classes registered in `RESEARCH_BLOCKS` / `RESEARCH_MODELS` and `presets()`.
- **Sequence-to-sequence core adapters (`src/hypercast4d/upstream_models.py`)**: Models placed under `src/hypercast4d/_vendor/tslib/models/<Name>.py` (with license/provenance recorded in `PROVENANCE.json`) and added to `UPSTREAM_MODELS` + `upstream_defaults()`.
  - If the backbone uses data-dependent Python control flow (such as `if tensor.shape[1] % k:` or dynamic FFT indexing) that standard `torch.fx` proxies cannot trace directly, register a pure functional helper kernel in `src/hypercast4d/graph_lowering.py` and dispatch to it inside `SourceTracer.lower_module()`.

---

## 4. Quality & Documentation Checks

Before opening a pull request, run the full verification suite from `hypercast4d/`:

```bash
# 1. Docstring lint (NumPy convention across all of src/hypercast4d/)
.venv/bin/ruff check src/hypercast4d/

# 2. Doctests + Python test suite
.venv/bin/python -m pytest --doctest-modules src/hypercast4d/__init__.py \
  src/hypercast4d/algebras.py src/hypercast4d/layers.py \
  src/hypercast4d/architecture.py src/hypercast4d/graph_architecture.py \
  src/hypercast4d/data.py src/hypercast4d/training.py
.venv/bin/python -m pytest

# 3. Frontend TypeScript + Vitest suite + production bundle
npm --prefix web test
npm --prefix web run build

# 4. Strict documentation build (fails on any broken cross-reference or missing symbol)
.venv/bin/mkdocs build --strict
```
