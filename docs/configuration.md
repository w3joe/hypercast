# Schemas, YAML & REST API Reference

This page documents HyperCAST's YAML/JSON configuration files, serialized model schemas (v1 and v2), environment variables, and local FastAPI HTTP endpoints.

---

## 1. Main Evaluation Configuration (`configs/evaluation.yaml`)

Used by `hypercast4d-run --config <file>` ([`hypercast4d.experiment.run`][hypercast4d.experiment.run]).

| Path | Type | Default | Description |
|---|---|---|---|
| `data.path` | `str` | `data/raw/paper_data.xlsx` | Path to a 4-series Excel workbook (first column optional dates). |
| `data.target_column` | `str` | `Copper` | Target series moved to column index `0`. |
| `data.train_fraction` | `float` | `0.70` | Chronological training row fraction ($> 0$). |
| `data.validation_fraction` | `float` | `0.15` | Chronological validation row fraction ($f_{\text{train}} + f_{\text{val}} < 1$). |
| `experiment.output_dir` | `str` | `results/evaluation` | Artifact output directory (`_quick` appended when `--quick` is passed). |
| `experiment.cells` | `list[{window, horizon}]` | `10/1, 20/5, 60/20` | Lookback window and forecast horizon pairs. |
| `experiment.seeds` | `list[int]` | `[7, 19, 31, 43, 59]` | Random seeds evaluated per cell and model. |
| `experiment.quick` | `{cell_count, seed_count, epochs}` | `{1, 1, 3}` | Truncation limits applied by `--quick`. |
| `tensorboard.enabled` | `bool` | `true` | Emit live epoch/HParams logs under `tensorboard.log_dir`. |
| `tensorboard.log_dir` | `str` | `runs/tensorboard` | Parent directory for timestamped TensorBoard runs. |
| `tensorboard.flush_seconds` | `int` | `5` | SummaryWriter flush cadence in seconds ($\ge 1$). |
| `training.optimizer` | `dict` | Adam (`lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7, amsgrad=false`) | Optimizer settings matching Keras 2.12 defaults. |
| `training.loss` | `mse \| mae \| huber` | `mse` | Scaled training/validation criterion. |
| `training.batch_size` | `int` | `32` | Training mini-batch size. |
| `training.evaluation_batch_size` | `int` | `256` | Inference batch size. |
| `training.epochs` | `int` | `50` | Maximum epochs per fit. |
| `training.shuffle` | `bool` | `true` | Shuffle training windows per epoch. |
| `training.early_stopping_patience` | `int \| null` | `null` | Consecutive non-improving epochs before stopping (`null` disables). |
| `training.early_stopping_min_delta` | `float` | `0.0` | Minimum validation-loss improvement required to reset patience. |
| `training.restore_best_weights` | `bool` | `false` | Restore best validation state dict upon completion. |
| `training.device` | `cpu \| auto \| mps \| cuda` | `cpu` | PyTorch compute device (`auto` probes CUDA → MPS → CPU). |
| `models.enabled` | `list[str]` | All 7 models | Subset of `persistence`, `linear`, `cnn`, `lstm`, `hyper_quaternion`, `hyper_coquaternion`, `hyper_cl11`. |
| `models.{cnn,lstm,hyper}` | `dict` | Paper defaults | `units`, `kernel_size` (CNN), `dense_before_pool`, `dense_after_pool`, `dense_units`, `activation`, `dropout`, `pool_size`. |

---

## 2. Paper Reproduction Configuration (`configs/paper_reproduction.yaml`)

Used by `hypercast4d-reproduce --config <file>` ([`hypercast4d.reproduction.run_reproduction`][hypercast4d.reproduction.run_reproduction]).

| Section | Key Fields | Description |
|---|---|---|
| `protocol` | `metric: mae`, `metric_scale: normalized` | Enforces the supplementary notebook's normalized-MAE model selection rule. |
| `hyper_algebra_behavior` | `force_quaternion` \| `declared` | `force_quaternion` reproduces the released notebook's hardcoded `quaternions` override while logging both `declared_algebra` and `effective_algebra`; `declared` passes each candidate's algebra through. |
| `data` | `path`, `scaler`, `input_orders`, `windows`, `horizons`, `holdout` | Global `[0, 1]` MinMax scaling, 4 windows (`10, 20, 40, 60`), 4 horizons (`1, 5, 10, 20`), and shuffled 20% holdout split (`random_state=42`). |
| `cross_validation` | `folds: 10`, `shuffle: true`, `random_state: null` | 10-fold `KFold` over the 80% partition. |
| `search_spaces` | `cnn` (160), `lstm` (160), `hyper` (576) | Complete Cartesian grids traversed in `AnalysisGoogle.ipynb` parameter-sort order. |

---

## 3. Playground & CLI Evaluation Presets

Normalized by [`normalize_evaluation`][hypercast4d.playground_runner.normalize_evaluation]:

| Preset | Default Cells (`window/horizon`) | Default Seeds | Epochs | Validation Folds (`train / val`) | Can Unlock Final Test? |
|---|---|---|---:|---|---|
| `quick` | `10/1` | `[7]` | `3` | `[(0.70, 0.15)]` | No |
| `standard` | `10/1`, `20/5`, `60/20` | `[7, 19, 31, 43, 59]` | `50` | `[(0.70, 0.15)]` | Yes (once per candidate hash) |
| `robust` | All 16 (`10,20,40,60` × `1,5,10,20`) | `[7, 19, 31, 43, 59]` | `50` | `[(0.55, 0.10), (0.65, 0.10), (0.75, 0.10)]` | Yes (once per candidate hash) |

---

## 4. Serialized Architecture Formats (YAML / JSON)

=== "Schema v1 (Sequential Pipeline)"

    ```yaml
    schema_version: 1
    name: Quaternion GRU Residual
    input:
      representation: differences
      feature_order: [0, 1, 2, 3]
    layers:
      - id: hyper
        type: hyper_dense
        params:
          units: 8
          algebra: quaternion
      - id: norm
        type: layer_norm
        params: {}
      - id: gru
        type: gru
        params:
          hidden_size: 16
          num_layers: 1
          dropout: 0.0
      - id: last
        type: last_state
        params: {}
    head:
      type: persistence_residual
      zero_initialize: true
    ```

=== "Schema v2 (Executable Graph)"

    ```yaml
    schema_version: 2
    revision: source-graph-1-tslib-4e938a1767106324dd753b2a44832bf870a0252e
    name: Edited Graph
    sources:
      s0: { ... }  # Normalized Schema v1 initialization recipe
    nodes:
      - id: inputs
        kind: source
        source_ref: { source: s0, node: inputs }
        params: {}
      - id: inserted_hyper
        kind: hyper_dense
        params:
          algebra: quaternion
          units: 1
          shape_mode: preserve
          bias: true
      - id: output
        kind: source
        source_ref: { source: s0, node: output }
        params: {}
    edges:
      - { source: inputs, target: inserted_hyper, port: x }
      - { source: inserted_hyper, target: ..., port: args/0 }
    groups:
      - { id: layer-core, label: tsmixer }
    output: output
    ```

---

## 5. Cloud Environment Variables (GCP)

See [GCP GPU Execution](gcp_gpu.md) for provisioning prerequisites.

| Variable | Required | Validation Rule |
|---|---|---|
| `HYPERCAST_GCP_PROJECT` | Yes | GCP project ID |
| `HYPERCAST_GCP_ZONE` | Yes | Compute Engine zone (for example `us-central1-a`) |
| `HYPERCAST_GCP_BUCKET` | Yes | Private GCS bucket name for ephemeral job staging |
| `HYPERCAST_GCP_SERVICE_ACCOUNT` | Yes | Worker VM service account email |
| `HYPERCAST_GCP_IMAGE` | Yes | Prepared CUDA + PyTorch VM image name |
| `HYPERCAST_GCP_IMAGE_PROJECT` | Yes | Project owning `HYPERCAST_GCP_IMAGE` |
| `HYPERCAST_GCP_SUBNET` | Yes | Private VPC subnet with Cloud NAT / Private Google Access |
| `HYPERCAST_GCP_MAX_HOURS` | No (`"6"`) | Integer `1..24`; sets `--max-run-duration` auto-delete ceiling |

---

## 6. Local Loopback REST API (`http://127.0.0.1:8765`)

When `hypercast4d-playground` (or `hypercast server start`) is running, interactive OpenAPI documentation is available at `/docs`.

| Method & Path | Purpose |
|---|---|
| `GET /health` | Liveness check (`{"ok": true}`). |
| `GET /api/v1/identity` | Workspace lock, PID, package version, and graph revision. |
| `GET /api/v1/catalog` | Layer catalog, algebras, 25 presets, 52-method index, and evaluation presets. |
| `GET /api/v1/compute` | Local, Modal (`modal token info`), and GCP (`gcloud auth`) capability status. |
| `POST /api/v1/architectures/validate` | Validate Schema v1 or v2 shapes, parameter count, and warnings at `(window, horizon)`. |
| `POST /api/v1/architectures/convert` | Convert a Schema v1 pipeline into a Schema v2 executable graph. |
| `POST /api/v1/architectures/describe` | Static port/parameter contracts for Schema v2 draft repair without shape validation. |
| `POST /api/v1/architectures/edit` | Apply `add`, `insert`, `set`, `replace`, or `remove` across evaluation `cells`. |
| `POST /api/v1/architectures/weights` | Deterministic seed-0 preview of `HyperDense` component & effective real matrices. |
| `GET /api/v1/architectures` | List saved local architecture records. |
| `POST /api/v1/architectures` | Save/update a local architecture record (`id`, `spec`, optional `view`). |
| `GET /api/v1/architectures/{id}` | Retrieve a saved architecture record by UUID. |
| `GET /api/v1/jobs` | List all persisted validation and final-test jobs. |
| `POST /api/v1/jobs` | Enqueue a validation job (`architecture`, `evaluation`, `execution`). |
| `GET /api/v1/jobs/{id}` | Retrieve job `status`, `request`, `summary`, `runs`, and `per_lead` tables. |
| `GET /api/v1/jobs/{id}/log` | Trailing plain-text worker log (`training.log`). |
| `POST /api/v1/jobs/{id}/cancel` | Cancel a queued or running job. |
| `POST /api/v1/jobs/{id}/final-test` | Unlock and enqueue the one-time held-out `final_test` run (`409` if already used). |
| `GET /api/v1/jobs/{id}/weights` | Trained `HyperDense` weight snapshots (`weights.json`). |
| `GET /api/v1/jobs/{id}/forecasts` | Up to 2,000 point forecasts filtered by `window`, `horizon`, `seed`, `fold`, `lead`. |
| `GET /api/v1/jobs/{id}/predictions.csv` | Download complete `predictions.csv` file. |
| `GET /api/v1/comparison-archives` | List audited read-only benchmark comparison archives. |
| `GET /api/v1/comparison-archives/{id}/forecasts` | Query up to 2,000 archived point forecasts for replay. |
