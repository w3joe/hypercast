# Experiment CLI

The `hypercast` CLI edits the same pipeline/graph formats and submits to the same
job service as the playground. This release supports individual experiments;
sweep orchestration is not included. macOS and Linux are supported.

## Install and discover

```sh
python -m pip install -e .
hypercast --help
hypercast model list
hypercast layer types
```

Activate the project's environment first (`source .venv/bin/activate`), or use
`.venv/bin/hypercast`. `python -m hypercast4d.cli` is equivalent. Existing
`hypercast4d-run`, `hypercast4d-reproduce`, and playground commands remain valid.

Commands accept `--json` for structured stdout (`job logs --follow --json` emits JSON Lines). Progress goes to stderr. Exit
codes: 0 success, 2 invalid input/declined confirmation, 3 backend unavailable,
4 failed/interrupted experiment, 130 cancelled experiment or stopped observation.

## Build a HyperDense-input experiment

```sh
hypercast model create tslib-itransformer --out experiment.yaml
hypercast layer insert experiment.yaml --before core --type dense --id input_lift --set units=12
hypercast layer insert experiment.yaml --before core --type hyper_dense --id input_hyper --set algebra=quaternion --set units=3
hypercast layer list experiment.yaml
hypercast model validate experiment.yaml --cells 10/1,20/5
hypercast experiment run experiment.yaml --preset standard --epochs 50 --batch-size 128 --seeds 7,19,31 --cells 10/1,20/5 --target local
```

For 2D use `algebra=complex, units=6`; for 3D use `algebra=tricomplex, units=4`;
for 4D use `algebra=quaternion, units=3`. Use separate `--set` options, not a
comma-separated assignment. All have 12 real output features. Four raw features
are not divisible by three, so the explicit 4→12 lift is necessary for 3D.
No projection or padding is silently inserted by the CLI.

This reproduces the HyperDense configuration and evaluation settings of the
recent input ablation. To compare against the original, create another preset
file and run it with the same flags. Layer edits are in-place and atomic; use
`--out other.yaml` to fork a model. Existing output paths are not overwritten.
Failed edits leave the original file untouched.

## Edit internals and connections

```sh
hypercast model expand experiment.yaml --out graph.yaml --cells 10/1,20/5
hypercast layer list graph.yaml --cells 10/1,20/5
# Substitute an actual node ID from the listing:
hypercast layer set graph.yaml NODE_ID --set out_features=24 --cells 10/1,20/5
hypercast layer replace graph.yaml NODE_ID --type hyper_dense --set algebra=complex --cells 10/1,20/5
hypercast graph connect graph.yaml --source SOURCE_ID --target TARGET_ID --port args/0 --replace
hypercast graph disconnect graph.yaml --target TARGET_ID --port args/0 --allow-invalid
hypercast graph output graph.yaml OUTPUT_NODE_ID
```

Pipeline IDs such as `core` are not automatically graph-node IDs. Inspect the
expanded graph first. Custom Dense nodes use `units`; source-bound `Linear`
nodes use the displayed `out_features` setting. HyperDense `units` (or source
`out_features`) are **hypercomplex** units, not real width. Listings show both.

`layer add` creates an unconnected graph operation (or appends a pipeline
layer). `layer insert --before ID --port PORT` inserts on one input edge;
`--after ID` requires a single outgoing edge. Multi-input operators must be
added and connected explicitly. `layer remove FILE ID` removes incident graph
edges, without silently joining branches; choose another output before
removing the forecast node. Pipeline removal retains its implicit sequence.

New graph HyperDense insertions default to `shape_mode=preserve`: units are inferred
from the input's last axis, with zero-padding and cropping to preserve its width.
Use `--set shape_mode=manual --set units=8` for explicit sizing. Existing graphs
without a mode retain manual behavior. To repair a HyperDense shape mismatch:
`hypercast layer set graph.yaml NODE_ID --set shape_mode=preserve --cells 10/1,20/5`.
Listings include the resolved units, padded width and cropped output width; these
are unknown until an invalid draft validates. Pipeline (v1) sizing is unchanged.

Dense/HyperDense replacement preserves width and bias, checks all supplied
cells, and refuses incompatible dimensions. Use `layer set` for intentional
width changes. Graph replacements detach the selected call from shared weights;
parameter changes on a shared source update its shared calls. Listings expose
the shared-call IDs. Shapes are evaluated at the first listed cell; validation
and replacements check every listed cell.

Drafts with incomplete connections or incompatible intermediate shapes require
`--allow-invalid`; malformed parameters and new cyclic connections are still
rejected. Drafts cannot run. Canvas view metadata is retained. Locked models
must be copied through `model create`; editing never changes the preset catalogue.

## Evaluation and execution

```sh
hypercast experiment run experiment.yaml --evaluation evaluation.yaml --epochs 50 --dry-run
hypercast experiment run experiment.yaml --target modal --gpu L4 --yes --detach
hypercast experiment run experiment.yaml --target gcp --gpu L4 --gpu-count 16 --yes --detach
hypercast job list
hypercast job status JOB_ID --follow
hypercast job logs JOB_ID --follow
hypercast job results JOB_ID --out exported-results
hypercast job cancel JOB_ID
hypercast experiment final-test VALIDATION_JOB_ID --yes
```

Evaluation YAML contains the existing evaluation object, not an architecture or
a main-paper runner config. CLI flags override `--evaluation`; `--eval key=value`
supports other existing settings, e.g. `--eval restore_best_weights=true`.
Unknown settings and custom folds are rejected. Defaults are the existing Quick
preset unless explicitly overridden. CPU is the default; use `--device auto`,
`mps`, or `cuda` locally. Cloud jobs force CUDA.

Dry-run validates all cells and prints shapes, parameter counts, normalized
configuration, trial count and resource topology. It starts no backend, creates
no job, uploads no data and does not prove cloud IAM/quota/capacity readiness.
Cloud execution requires interactive confirmation or `--yes`; use the existing
[GCP setup guide](gcp_gpu.md) or Modal setup. Sixteen L4s uses two 8-GPU VMs;
A100 40GB counts stop at eight. Multiple GPUs parallelize independent trials,
not one model. Configured cloud runtime and cleanup safeguards still apply.

Run waits by default; `--detach` returns immediately. Ctrl+C stops **observation**,
not the submitted job: use `job cancel` explicitly. A detached backend keeps
running after the terminal exits. Cancellation follows the same path as the UI.
Final-test requires a completed non-Quick validation job and respects the existing
test-once guard; validation runs never automatically evaluate held-out test data.

## Shared backend and UI

```sh
hypercast server start --port 8765
hypercast server status --json
hypercast model save experiment.yaml
hypercast model export ARCHITECTURE_ID --out exported.yaml
hypercast4d-playground
hypercast server stop
```

Experiments start/reuse a loopback backend automatically. Use the URL printed at
submission or by `server status` to open its UI; auto-start uses an available
ephemeral port unless a server was started with `--port`. The playground launcher
reuses an existing backend for the same workspace.

The project defaults to the current directory and results to
`PROJECT/results/playground`. Set `--project-root` and `--results-root` explicitly
when running elsewhere; supply the same workspace flags for later job commands.
Explicit model/evaluation/output file paths are relative to your shell, while
dataset paths are relative to the project. Dataset files must resolve inside
the project's `data/` directory. Runs register a model snapshot and appear in
the UI's Runs/Compare views. `model save` registers a model without training.

An OS file lock prevents multiple updated backends from owning/recovering the
same queue. Discovery metadata and logs are under `RESULTS/.runtime/`. Stale
records are not trusted without an identity check. An old, unregistered server
on the default port must be stopped and restarted first; do not mix old server
versions on custom ports with the same results directory. This CLI does not
support remote or multi-user servers. `server stop` refuses active/queued jobs.

Job inspection uses the running backend; if it has been stopped, start it again
for that workspace. Recovery marks abandoned running jobs interrupted and
resumes queued jobs, matching the UI. It does not automatically restart failed
or interrupted training or recover orphaned cloud resources.
