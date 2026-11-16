# Supported Components & Models

This reference catalogs every input representation, layer block, Schema v2 graph operator, forecast head, and preset supported by HyperCAST.

---

## 1. Input Representations (`spec["input"]`)

| Option | Values / Bounds | Default | Behavior |
|---|---|---|---|
| `representation` | `levels`, `centered`, `differences` | `levels` | `levels` passes scaled observations $[B, W, C]$ unchanged; `centered` subtracts the final observed step $X[:, -1:, :]$; `differences` computes first differences $X[:, 1:] - X[:, :-1]$ of shape $[B, W-1, C]$. |
| `feature_order` | Non-empty unique subset of `0..features-1` | `[0, 1, 2, 3]` | Selects and orders input channels fed into the encoder; index `0` is always the forecast target channel. |

---

## 2. Pipeline Layer Blocks (Schema v1 & v2)

All blocks below are available in Schema v1 `layers[]` (`architecture.py`), and the neural/processing/reduction blocks are also available as standalone Schema v2 graph nodes (`graph_architecture.py`).

### Feature Mixing

| `type` / `kind` | Parameters (Default, Bounds) | Input → Output Shape | Notes |
|---|---|---|---|
| `dense` | `units` (`32`, `1..512`), `bias` (`true` in v2) | `[..., C_in] -> [..., units]` | Standard `torch.nn.Linear`. Supports `"units": "horizon"` on Schema v2 readout nodes. |
| `hyper_dense` | `units` (`8`, `1..512//d`), `algebra` (`"quaternion"`), `bias` (`true` in v2), `shape_mode` (`"manual"` or `"preserve"` in v2) | Manual: `[..., d*n] -> [..., d*units]`<br>Preserve (v2): `[..., C] -> [..., C]` | Uses [`HyperDense`][hypercast4d.layers.HyperDense] (`manual`) or [`ShapePreservingHyperDense`][hypercast4d.layers.ShapePreservingHyperDense] (`preserve`). |

### Temporal & Recurrent Blocks

| `type` / `kind` | Parameters (Default, Bounds) | Input → Output Shape | Notes |
|---|---|---|---|
| `causal_conv` | `channels` (`16`, `1..512`), `kernel_size` (`3`, `1..31`), `dilation` (`1`, `1..64`) | `[B, T, C_in] -> [B, T, channels]` | Left-padded causal 1D convolution; adds `dilation * (kernel_size - 1)` to receptive field. |
| `tcn` | `channels` (`16`, `1..512`), `kernel_size` (`3`, `1..31`), `dilations` (`[1, 2, 4, 8]`, `1..8` items in `1..64`), `dropout` (`0.1`, `0..0.95`) | `[B, T, C_in] -> [B, T, channels]` | Residual causal TCN with `SiLU` + `Dropout` + `LayerNorm` per block. |
| `lstm` | `hidden_size` (`16`, `1..512`), `num_layers` (`1`, `1..3`), `dropout` (`0.0`, `0..0.95`) | `[B, T, C_in] -> [B, T, hidden_size]` | Full-sequence LSTM; freezes redundant `bias_hh` to match Keras parameter counts. |
| `gru` | `hidden_size` (`16`, `1..512`), `num_layers` (`1`, `1..3`), `dropout` (`0.0`, `0..0.95`) | `[B, T, C_in] -> [B, T, hidden_size]` | Full-sequence GRU. |

### Processing & Sequence Reduction

| `type` / `kind` | Parameters (Default, Bounds) | Input → Output Shape | Notes |
|---|---|---|---|
| `layer_norm` | None | `[..., C] -> [..., C]` | `torch.nn.LayerNorm(C)` over the feature dimension. |
| `activation` | `kind` (`"relu"`, one of `relu`, `gelu`, `silu`, `tanh`, `linear`) | `[...] -> [...]` | Pointwise activation (`linear` maps to `nn.Identity`). |
| `dropout` | `p` (`0.1`, `0.0..0.95`) | `[...] -> [...]` | `torch.nn.Dropout(p)`. |
| `max_pool` | `size` (`2`, `1..64`), `stride` (`size`, `1..64`) | `[B, T, C] -> [B, T', C]` | Temporal 1D max pooling where $T' = \lfloor(T - \text{size})/\text{stride}\rfloor + 1$ (v1 pipeline). |
| `last_state` | None | `[B, T, C] -> [B, C]` | Selects final step `inputs[:, -1]`. |
| `mean_pool` | None | `[B, T, C] -> [B, C]` | Averages across time `inputs.mean(dim=1)`. |
| `flatten` | None | `[B, T, C] -> [B, T*C]` | Flattens all axes after batch (`start_dim=1`). |

### Research Mechanism Blocks (Schema v1)

| `type` | Parameters (Default, Bounds) | Input → Output Shape | Notes |
|---|---|---|---|
| `decomposition` | `kernel_size` (`25`, odd `1..101`) | `[B, T, C] -> [B, T, 2*C]` | Moving-average seasonal/trend split concatenated along channels. |
| `temporal_projection` | `output_steps` (`8`, `1..512`) | `[B, T, C] -> [B, output_steps, C]` | Separate learned time-axis projection per channel. |
| `patch_embedding` | `d_model` (`32`, `1..512`), `patch_size` (`4`, `1..128`), `stride` (`2`, `1..patch_size`) | `[B, T, C] -> [B, P, d_model]` | Joint multivariate patch tokens + positional embeddings. |
| `temporal_attention` | `d_model` (`32`, `4..256`), `n_heads` (`4`, `1..16`, divides `d_model`), `num_layers` (`2`, `1..4`), `dropout` (`0.1`, `0..0.95`) | `[B, T, C] -> [B, T, d_model]` | Transformer encoder along time/patch axis. |
| `variable_attention` | Same as `temporal_attention` | `[B, T, C] -> [B, T, C]` | Transformer encoder across variables projected back to `T` steps. |

---

## 3. Graph-Only Structural & Branch Operators (Schema v2 `NEW_OPS`)

When editing an expanded Schema v2 graph, six additional structural operators are available:

| `kind` | Ports | Parameters (Default) | Semantics |
|---|---|---|---|
| `add` | `a`, `b` | `{}` | Elementwise residual addition `a + b` (shapes of `a` and `b` must match exactly). |
| `multiply` | `a`, `b` | `{}` | Elementwise gating product `a * b` (shapes of `a` and `b` must match exactly). |
| `concat` | `a`, `b` | `dim` (`-1`) | `torch.cat([a, b], dim=dim)`. |
| `reshape` | `x` | `shape` (`[0, -1]`, 1–8 ints) | `0` copies the corresponding input axis; at most one `-1` infers the remaining size. |
| `permute` | `x` | `dims` (`[0, 2, 1]`) | Axis permutation listing each input axis `0..ndim-1` once. |
| `softmax` | `x` | `dim` (`-1`) | `torch.softmax(x, dim=dim)`. |

---

## 4. Forecast Heads (`spec["head"]`)

| `type` | Default `zero_initialize` | Formula ($\text{anchor} = X[:, -1, 0:1]$) |
|---|---|---|
| `direct` | `False` | $\hat{Y} = \text{Linear}(z)$ |
| `persistence_residual` | `True` | $\hat{Y} = \text{anchor} + \text{Linear}(z)$ (starts exactly at persistence when zero-initialized) |
| `cumulative_residual` | `True` | $\hat{Y} = \text{anchor} + \operatorname{cumsum}(\text{Linear}(z), \text{dim}=-1)$ |

---

## 5. TSLib-15 Sequence Adapters (`tslib_*`)

HyperCAST wraps 15 pinned MIT-licensed TSLib models inside [`UpstreamSequence`][hypercast4d.upstream_models.UpstreamSequence] (`[B, T, C] -> [B, T, C]`):

- **Left-replication padding**: History length $T$ is padded on the left to $T_{\text{pad}} = \max(32, \lceil T / 32 \rceil \cdot 32)$, predicted over $T_{\text{pad}}$ steps, and sliced back to the trailing $T$ steps before feeding downstream layers/readouts.
- **Editable core parameters** (`d_model` in `8..256` divisible by 8 and `n_heads`, `num_layers` in `1..3`, `n_heads` in `1..16`, `dropout` in `0..0.95`):
  - `tslib_patchtst`, `tslib_itransformer`, `tslib_crossformer`, `tslib_msgnet`, `tslib_timexer`: `d_model`, `num_layers`, `n_heads`, `dropout`
  - `tslib_timesnet`, `tslib_timemixer`, `tslib_tsmixer`: `d_model`, `num_layers`, `dropout`
  - `tslib_micn`, `tslib_segrnn`: `d_model`, `dropout`
  - `tslib_lightts`: `d_model`
  - `tslib_dlinear`, `tslib_frets`, `tslib_film`, `tslib_scinet`: fixed core settings (editable via Schema v2 graph expansion or surrounding layers)

---

## 6. Built-In Presets (25 Total)

Call [`presets()`][hypercast4d.architecture.presets] or `hypercast model list` to inspect all 25 presets:

- **Locked Figure 3 Paper Models (5)**: `paper-cnn`, `paper-lstm`, `paper-quaternion`, `paper-coquaternion`, `paper-cl11`
- **Editable Native Templates (2)**: `residual-tcn`, `quaternion-gru-residual`
- **Editable Mechanism-Inspired Templates (3)**: `research-dlinear`, `research-patchtst`, `research-itransformer`
- **Editable TSLib-Core Templates (15)**: `tslib-dlinear`, `tslib-patchtst`, `tslib-itransformer`, `tslib-timesnet`, `tslib-crossformer`, `tslib-film`, `tslib-frets`, `tslib-lightts`, `tslib-micn`, `tslib-msgnet`, `tslib-scinet`, `tslib-segrnn`, `tslib-timemixer`, `tslib-timexer`, `tslib-tsmixer`
