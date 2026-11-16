# Algorithms & Mathematics

This guide documents the mathematical formulation of HyperCAST's algebras, hypercomplex layers, symbolic graph lowering, leakage-safe data partitioning, and point-forecast diagnostics.

---

## 1. Finite-Dimensional Real Algebras

An algebra $\mathcal{A}$ of dimension $d$ over $\mathbb{R}$ with ordered basis $\{e_0, e_1, \dots, e_{d-1}\}$ (where $e_0 = 1$ is the multiplicative identity) is completely specified by its cubic structure-constant tensor $C \in \mathbb{R}^{d \times d \times d}$:

$$e_a e_b = \sum_{c=0}^{d-1} C_{a, b, c}\, e_c.$$

Given two operands $u, v \in \mathbb{R}^{\dots \times d}$ storing basis coordinates along their final axis, [`Algebra.multiply`][hypercast4d.algebras.Algebra.multiply] evaluates their bilinear product via Einstein summation:

$$(u \cdot v)_{\dots, c} = \sum_{a=0}^{d-1} \sum_{b=0}^{d-1} u_{\dots, a}\, v_{\dots, b}\, C_{a, b, c}.$$

### Supported Algebras in `hypercast4d.algebras`

| Key | Aliases | Dim ($d$) | Commutative | Associative | Defining Relations ($e_0 = 1$) |
|---|---|---:|---|---|---|
| `complex` | `complexes` | 2 | Yes | Yes | $e_1^2 = -1$ |
| `split_complex` | `split-complex`, `split complex` | 2 | Yes | Yes | $e_1^2 = +1$ |
| `tricomplex` | `trinion`, `trinions`, `tricomplexes` | 3 | Yes | Yes | Cyclic $\mathbb{R}[h]/(h^3 - 1)$: $e_1^2 = e_2,\; e_1 e_2 = e_2 e_1 = 1,\; e_2^2 = e_1$ |
| `quaternion` | `quaternions` | 4 | No | Yes | $e_1^2 = e_2^2 = e_3^2 = -1,\; e_1 e_2 = e_3,\; e_2 e_3 = e_1,\; e_3 e_1 = e_2$ |
| `coquaternion` | `coquaternions` | 4 | No | Yes | $e_1^2 = -1,\; e_2^2 = e_3^2 = +1,\; e_1 e_2 = e_3,\; e_2 e_3 = -e_1,\; e_3 e_1 = e_2$ |
| `cl11` | `cl(1,1)`, `clifford11` | 4 | No | Yes | $e_1^2 = +1,\; e_2^2 = -1,\; e_3^2 = +1,\; e_1 e_2 = e_3,\; e_1 e_3 = e_2,\; e_2 e_3 = e_1$ |
| `octonion` | `octonions` | 8 | No | No (alternative) | $e_k^2 = -1$ ($k=1\dots 7$); oriented Fano triples $(1,2,3)$, $(1,4,5)$, $(1,7,6)$, $(2,4,6)$, $(2,5,7)$, $(3,4,7)$, $(3,6,5)$ |

!!! note "4D Multiplication Tables"
    The 4D `quaternion`, `coquaternion`, and `cl11` structure constants match Table 1 of Kycia & Niemczynowicz (2025), [*4D hypercomplex-valued neural network in multivariate time series forecasting*](https://doi.org/10.1038/s41598-025-08957-5). Because octonion sign conventions vary across the literature, HyperCAST pins the Baez Fano-plane orientation so saved checkpoints remain bit-reproducible.

---

## 2. `HyperDense` Layer & Block-Matrix Equivalence

### Component-Major Layout

[`HyperDense(in_features=n, out_features=m, algebra=A)`][hypercast4d.layers.HyperDense] accepts real tensors of shape $(*, d \cdot n)$ arranged in **component-major order**:

$$x = \big[\underbrace{x^{(0)}_1, \dots, x^{(0)}_n}_{e_0\text{ (real)}},\; \underbrace{x^{(1)}_1, \dots, x^{(1)}_n}_{e_1},\; \dots,\; \underbrace{x^{(d-1)}_1, \dots, x^{(d-1)}_n}_{e_{d-1}}\big].$$

Let $W \in \mathbb{R}^{d \times n \times m}$ denote the $d$ trainable real component weight matrices $W^{(0)}, \dots, W^{(d-1)}$ and $b \in \mathbb{R}^{d \times m}$ the optional bias. Using input-on-the-left multiplication ($x \cdot W$), output component $c \in \{0, \dots, d-1\}$ is:

$$y^{(c)} = \sum_{b=0}^{d-1} \sum_{a=0}^{d-1} C_{b, a, c}\, x^{(b)} W^{(a)} + b^{(c)}.$$

Equivalently, `HyperDense` acts as an ordinary real matrix multiplication $y = x W_{\text{eff}} + \text{vec}(b)$ where the $(d n) \times (d m)$ block matrix $W_{\text{eff}}$ has $(b, c)$-th block

$$(W_{\text{eff}})_{b, c} = \sum_{a=0}^{d-1} C_{b, a, c}\, W^{(a)} \in \mathbb{R}^{n \times m}.$$

For example, for Hamilton quaternions ($d = 4$), $W_{\text{eff}}$ is the structured $4 \times 4$ block operator:

$$W_{\text{eff}}^{\mathbb{H}} = \begin{bmatrix}
W^{(0)} & W^{(1)} & W^{(2)} & W^{(3)} \\
-W^{(1)} & W^{(0)} & -W^{(3)} & W^{(2)} \\
-W^{(2)} & W^{(3)} & W^{(0)} & -W^{(1)} \\
-W^{(3)} & -W^{(2)} & W^{(1)} & W^{(0)}
\end{bmatrix}.$$

### Parameter Complexity & Initialization

| Layer | Real Input Width | Real Output Width | Kernel Parameters | Bias Parameters |
|---|---:|---:|---:|---:|
| `nn.Linear(d*n, d*m)` | $d n$ | $d m$ | $d^2 n m$ | $d m$ |
| `HyperDense(n, m, algebra)` | $d n$ | $d m$ | $d n m$ ($1/d\times$) | $d m$ |

By default (`reset_parameters`), each component slice $W^{(a)} \in \mathbb{R}^{n \times m}$ is initialized independently with Glorot/Xavier normal ($\mathcal{N}(0, \frac{2}{n + m})$) and $b = 0$, matching the reference Keras `HyperDense` implementation. Controlled variance-matched initialization protocols used in internal ablation studies are documented in [`hypercast4d.experiment_controls`][hypercast4d.experiment_controls].

---

## 3. Auto-Fit Dimension Adaptation (`ShapePreservingHyperDense`)

When inserting a hypercomplex layer onto an existing connection whose last-axis width $C$ is not a multiple of $d$ (or when preserving exact downstream shapes without manual unit arithmetic), [`ShapePreservingHyperDense`][hypercast4d.layers.ShapePreservingHyperDense] performs:

1. **Unit inference**: $u = \lceil C / d \rceil$, $C_{\text{pad}} = u \cdot d$, $\Delta = C_{\text{pad}} - C$.
2. **Zero-padding**: $x_{\text{pad}} = [x,\; 0_{\dots \times \Delta}] \in \mathbb{R}^{* \times C_{\text{pad}}}$.
3. **Square hypercomplex map**: $z = \text{HyperDense}(u, u, \mathcal{A})(x_{\text{pad}})$.
4. **Contiguous crop**: $y = z[\dots, :C]\text{.contiguous()} \in \mathbb{R}^{* \times C}$.

Calling `.contiguous()` after slicing guarantees compact memory strides for downstream TSLib blocks that invoke `.view()` instead of `.reshape()`.

---

## 4. Symbolic FX Lowering vs. Executable Graph Execution

HyperCAST supports two interoperable model representations:

- **Schema v1 (`ExperimentalForecaster`)**: A sequential pipeline (`input` → `layers[]` → `head`) with analytical shape and receptive-field tracking.
- **Schema v2 (`GraphForecaster`)**: An explicit port-addressed directed acyclic graph (DAG) lowered from one or more Schema v1 recipes via [`lower_source`][hypercast4d.graph_lowering.lower_source].

Key design guarantees in `graph_lowering.py` and `graph_architecture.py`:

- **Symbolic lowering, never value tracing**: Data-dependent FFT period selection (`TimesNet`, `MSGNet`), odd/even tree splitting (`SCINet`), segment merging (`Crossformer`), and reversible instance normalization (`StandardNorm`) are lowered into registered pure functions (`period_select`, `period_fold`, `interleave`, etc.) while keeping every trainable `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, and attention projection exposed as an editable `call_module` node.
- **Functionalized mutations & live `training` flag**: In-place methods (`masked_fill_`, `add_`, etc.) are rewritten to out-of-place operations so dataflow edges are explicit, and `F.dropout` reads the live `GraphForecaster.training` attribute.
- **Deterministic Kahn scheduling & dead-branch pruning**: `GraphForecaster._order()` topologically sorts the DAG using node array order as a deterministic tie-breaker and prunes any node not reachable backward from `spec["output"]`.

---

## 5. Leakage-Safe Chronological Splits (`chronological-v1`)

Given a multivariate table of $N$ chronological rows, lookback window $W$, forecast horizon $H$, training fraction $f_{\text{train}}$, and validation fraction $f_{\text{val}}$:

- Row boundaries are $N_{\text{train}} = \lfloor N \cdot f_{\text{train}} \rfloor$ and $N_{\text{val}} = \lfloor N \cdot (f_{\text{train}} + f_{\text{val}}) \rfloor$.
- [`MinMaxStats`][hypercast4d.data.MinMaxStats] fits $\min$ and $\max - \min$ **strictly on rows $[0, N_{\text{train}})$**.
- For each candidate target start index $t \in [W, N - H]$, the sample $(X_{t-W:t},\; Y_{t:t+H, 0})$ is assigned by its **entire target span** $[t, t + H)$:
  - **Train**: $t + H \le N_{\text{train}}$
  - **Validation**: $t \ge N_{\text{train}}$ and $t + H \le N_{\text{val}}$
  - **Test**: $t \ge N_{\text{val}}$

Windows whose target horizons straddle $N_{\text{train}}$ or $N_{\text{val}}$ are **discarded**, ensuring zero target overlap across splits while allowing validation and test lookback histories $X_{t-W:t}$ to condition on prior observed values.

---

## 6. Point-Forecast Diagnostics

In addition to raw-unit `mae`, `mse`, and persistence ratios (`mae / persistence_mae`), [`hypercast4d.diagnostics`][hypercast4d.diagnostics] computes five lead-resolved diagnostics for each forecast lead $h \in \{1, \dots, H\}$ using origin price $p_t = y_{t-1} > 0$:

1. **Relative returns**: $r^{\text{act}}_{t,h} = \frac{y_{t+h-1} - p_t}{p_t}$ and $r^{\text{pred}}_{t,h} = \frac{\hat{y}_{t+h-1} - p_t}{p_t}$.
2. **Directional accuracy** (`directional_accuracy` vs. `direction_baseline_accuracy`): 3-class direction $\operatorname{dir}(r) \in \{-1, 0, +1\}$ with flat deadband $|r| \le 10^{-6}$, compared against the modal $h$-step training direction (tie-broken toward $-1 < 0 < +1$).
3. **Return correlation** (`return_correlation`): Pearson correlation between $r^{\text{pred}}_{\cdot,h}$ and $r^{\text{act}}_{\cdot,h}$, reported only when $\ge 2$ valid pairs exist and both series have peak-to-peak spread $> 10^{-6}$.
4. **Signed bias** (`bias`): $\frac{1}{M}\sum_t (\hat{y}_{t+h-1} - y_{t+h-1})$ in original target units.
5. **Tail & large-move error** (`p95_abs_error`, `large_move_mae`, `large_move_mae_ratio`): 95th percentile of $|\hat{y} - y|$, and MAE conditioned on $|r^{\text{act}}_{t,h}| > q_{0.90}(|r^{\text{train}}_h|)$ fitted strictly on $h$-step pairs inside the training partition.

Undefined ratios, empty large-move subsets, and degenerate correlations are emitted as `None` / `null` (never zero-filled).
