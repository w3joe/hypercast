# Graph-native architecture playground

The Builder uses a React Flow node editor with labelled operation cards and
visible directed connections. Stages expand into their real executable operations.
Cards show exact validated tensor shapes; their dimensions do not imply tensor
proportions. ELK arranges stages and operations from left to right, routing branches
through separate lanes. The renderer never changes graph IDs, ports, initialization
sources or execution. This replaces the experimental Three.js slab diagram.

The default overview reads like a model figure: input preparation and the forecast
readout form stages, and adjacent activations/dropout are included with their
parent layers when there is no branching boundary. Flattening is included with
the following stage. For example, Paper CNN shows six stages for its 21 traced
operations; TSMixer shows Input → TSMixer → Forecast. These are presentation-only
partitions. Expand a stage or search the tree to reach every original operation.
Native unused indexing helpers are tucked into their stages; disconnected edits
and independent duplicates remain visible in the disconnected area.

## Experiment workflow

The collapsible, resizable left panel has four sections: **Architecture** (presets,
saved models, searchable stage/layer tree, import/export and save), **Inspect**
(settings, named inputs, replacement, weights and selected connection), **Add**
(layer catalog and insertion destination), and **Run** (evaluation and compute).
Click a node to select it and reveal Inspect. Select a tree entry to expand its
stage and focus the camera. Shift-click or use tree checkboxes for multi-selection.
**Find layer** (Cmd/Ctrl K) opens and focuses search from any Builder panel, including
when the sidebar is hidden. The tree appears before model file controls and defaults
to layers; **Show tensor operations** or a search exposes all executable operations.
A selected stage lists its layers first, with tensor plumbing in a disclosure.
The layer inspector provides previous/next, a layer picker and **Locate on canvas**;
essential parameter fields appear above Dense/HyperDense conversion options.

Pan by dragging the background; pinch to zoom. Undo, redo, zoom, fit and reset are
standalone icon buttons at the bottom right. Reset restores the collapsed stage
overview and fits it without modifying the executable graph. Camera and selection
survive Builder / Runs / Compare navigation. On narrow screens the sidebar overlays
the diagram. Larger diagrams include a clickable overview map and a zoom indicator.
The editor uses HTML and SVG, so it does not require WebGL.

Nodes stay fixed in an automatic layout; pan by dragging the background. Every
connection has an arrow, including parallel input ports and connections to drafts.
Click **+** on an arrow to insert a layer and wire both sides automatically. Markers
are visible at readable zoom levels and in Add mode. The Add palette requires a
chosen connection; creating an unconnected layer is an explicit **Advanced** action.
Select an existing layer and change **Layer type** to replace it in one undoable edit.
Replacement keeps the operation's sequence position, group, outgoing branches and
input arity; named ports are retained or mapped to the new layer's input contract.
Named source selectors in Inspect handle connections without dragging wires.
Multi-input layers must be added separately and wired through their named inputs.
Selecting a connection opens its source, destination and port in Inspect. The
selected layer's immediate connections are highlighted; unrelated nodes remain
selectable and appear muted.

### Automatic HyperDense insertion

New graph HyperDense layers enable **Auto-fit connection** in Inspect. They keep
the incoming tensor shape at model boundaries and internal real floating-point
connections. For last-axis width `C` and algebra dimension `d`, construction uses
`ceil(C / d)` units, appends zeros to reach the next multiple of `d`, applies
HyperDense in component-major order, then crops the output back to `C`. Quaternion
width 4 uses one unit; width 7 pads to 8, uses two units, and crops to 7.

Inspect shows the axis, inferred units, padded width and crop; the weight inspector
shows the padded matrix and explains the adaptation. Padding and cropping have no
learned parameters. This preserves shape, not values: the inserted layer is trained
with the rest of the model. Moving it recalculates its dimensions for the new edge.

The graph-only parameter `shape_mode: preserve` enables this behavior through the
canvas and CLI/API insertion. `shape_mode: manual` (or an absent mode in an existing
graph) retains explicit units and strict divisibility. In auto-fit mode, a saved
`units` value is only retained for switching back to manual sizing; execution derives
units from the actual connection. Existing invalid custom HyperDense drafts can be
repaired by enabling Auto-fit without a prior successful validation. Edits are undoable.

All selected evaluation cells must validate before running. Tuples, scalars, integer
indices, complex FFT values and tensors without batch/feature axes are unsupported
placements. A constructed layer's width is fixed; a data-dependent width change
produces an error identifying the node and input shapes. Every other tensor axis is
preserved, and residual joins and branch connections are unchanged. V1 models retain
their existing behavior.

### Swap Dense and HyperDense

Select a layer (or use **Find a layer**) and choose its **Layer type** in the
settings panel. Choose the target algebra first; Dense → HyperDense preserves
the real output width while converting it to the corresponding number of
hypercomplex units. For example, 24 real output features become 12 complex,
8 tricomplex, 6 quaternion, or 3 octonion units. Switching back restores the
equivalent real output width.

The switch checks every selected evaluation cell before committing, preserves
connections, group placement and bias, and rejects incompatible widths without
changing the graph. Input and output widths must both be divisible by the
selected algebra's component count. Changing a manually sized HyperDense algebra also
preserves its expanded real width or is rejected. No padding/projection is
silently inserted. Graph HyperDense supports vector, sequence and higher-rank
feature tensors; the v1 sequence-only contract is unchanged. Replacement weights
are newly initialized; a shared call becomes independent when switched. Undo
restores the original node.

| Dimension | Algebra choices | Basis/product convention |
| --- | --- | --- |
| 2D | Complex, split-complex | `i² = -1` or `i² = +1` |
| 3D | Cyclic tricomplex | `h² = k`, `k² = h`, `hk = 1` |
| 4D | Quaternion, coquaternion, `Cl(1,1)` | Paper multiplication tables |
| 8D | Octonion | Explicit Fano-plane orientation |

The cyclic tricomplex algebra is associative and commutative but has zero
divisors. Octonion multiplication is non-associative. HyperDense uses one binary
input-by-weight product, so both remain well-defined for this layer.

1. Load a preset; locked baselines retain **Clone to edit**. Expand a stage or
   search its operations from the Architecture tree.
2. Edit fields in Inspect. Text and numeric fields commit on blur or Enter;
   toggles and selects commit immediately. Invalid drafts remain editable and
   visibly distinguish stale shapes. Run is disabled until validation succeeds.
3. In Add, choose a connection marker or an explicit insertion destination.
   Choosing a unary layer atomically splits that edge, retaining its destination
   port. **Move selected layer** in Advanced tools uses the same destination workflow and reconnects
   the old neighbors only for unambiguous unary moves. Multi-input layers,
   branching moves and forecast endpoints require explicit named input wiring.
   **Advanced → Create unconnected layer** allows a disconnected draft. Selecting a line
   exposes its source, destination port, insertion and removal controls.
4. **Add another model** creates an independent subgraph. Connect its `Model input`
   and join its result or choose it as the forecast output. Disconnected drafts
   appear in a separate, labeled area.
5. Multi-select to duplicate independently, delete, group or share weights. Advanced
   tools retain independent/shared weight controls and forecast-output selection.
   Deletion removes incident connections and leaves required inputs visibly missing;
   it never silently reconnects former neighbors. Replacements preserve input arity.
6. Save or export/import YAML/JSON. View metadata is versioned as `renderer: flow`,
   `version: 1`, with empty `positions`, `expandedStages` and camera pan/zoom. The saved graph retains
   its original groups; summary partitions are derived again on load. Legacy Three.js camera coordinates
   and previous manual node positions are ignored; those models open automatically arranged and refitted. Valid stage expansion state is retained.
   View metadata does not change schema version 2 or executable graph semantics.
7. The HyperDense weight inspector opens over the diagram and preserves the
   selection and camera when closed. Training is allowed only after all selected
   evaluation cells pass; the server repeats validation before queueing execution.

## Execution contract

Version 2 stores `nodes`, input-port `edges`, `output`, initialization `sources`,
and a pinned lowering `revision`. Groups and saved `view` camera/collapse
state affect presentation only. Node-array order is the deterministic tie-break
between independent branches, including dropout calls, and is part of the hash.

The compiler symbolically lowers trusted source recipes into individual
operators. Only the persisted graph's reachable nodes and edges execute;
the original whole-model forward is not an execution fallback. No arbitrary
Python expressions, imports, or user-supplied callable names are evaluated.
The recipes preserve initialization and window-dependent constants. Explicit
module/state references preserve shared weights, including raw FFT parameters.
Nonlearned tensor state is registered as buffers, not trainable parameters.

Period selection, fold/unfold and segment-layout operations run on actual inputs
at execution time. They are not frozen from a sample trace. Recurrent and
spectral kernels remain named operations; ordinary trainable projections and
convolutions around them remain exposed. Structural operators with no settings
can be rewired or replaced rather than edited through arbitrary Python code.
Expected ports are available even for an invalid draft via the describe API.

V1 loading, execution, hashing and checkpoint behavior are unchanged. Conversion
is in memory, preserves existing internal overrides, and the UI clears the old
record ID so saving creates a separate v2 record. V2 checkpoints are compiled
from the same graph before restoring their state dictionaries.

The TSLib adapter still pads to a multiple of 32, predicts a same-length latent
sequence, crops it, then uses the playground readout. This refactor does not turn
the adapters into reproductions of published benchmark protocols, and does not
implement the remaining reference-only methods in the collection.

## Verification

`tests/test_graph_architecture.py` covers all 15 cores at windows 2/10/33,
horizons 1/3/7 and feature subsets, comparing source versus graph outputs,
input/parameter gradients, an Adam update, seeded training dropout, and exact
checkpoint round trips. It also checks every legacy preset, nondefault core
settings, internal overrides, topology edits, shared/independent weights,
hybrid model branches, dynamic period changes, invalid edges and draft repair.

`tests/test_playground.py` covers migration, saved views and all-cell server
validation. The frontend suite covers executable edits, navigation, undo/redo,
invalid-cell run gating, compute controls, deterministic ELK layout, collapsed
boundary ports, branch routing, disconnected members and legacy view migration.
Overview tests check linear-stage combinations, native scaffolding, preserved
branches, visible disconnected edits and complete expansion back to the real graph.
Component interaction tests cover accessible cards without WebGL.

`scripts/check_canvas.py` runs real Chromium checks for node selection, fixed placement,
connections, selection-to-panel synchronization, pan/zoom/fit/reset, sidebar resizing,
navigation, weight inspection, representative presets and a fully expanded transformer.
It compares validation metadata and parameter counts with and without node view
metadata. It also checks passive connection handles, combined-stage duplication/deletion,
canonical graph saving, legacy camera migration, summary/camera import round trips
and the narrow-screen sidebar.
`scripts/check_canvas_insertion.py` checks guided insert/move, exact input ports,
undo/redo, deletion, disconnected drafts, reconnection, replacement, field commits,
invalid-draft run gating and versioned save serialization. Save requests are
intercepted and training requests are blocked. Neither script launches training. `scripts/check_canvas_usability.py` checks search,
keyboard access, previous/next navigation, contextual insertion, branch choice,
minimap navigation and preservation of graph semantics after undo.

Run with a locally running playground and an installed Playwright Chromium:

```sh
python scripts/check_canvas.py http://127.0.0.1:8765 --browser-binary /path/to/chromium
python scripts/check_canvas_insertion.py http://127.0.0.1:8765 --browser-binary /path/to/chromium
```

### Simplified evaluation comparison

Compare uses three main charts: a sorted mean/range plot, scores across recorded
window/horizon settings. When a group has one setting with multiple forecast leads,
the second chart shows error by lead instead. The third plots mean error against
total model parameters with a log axis (linear when zero-parameter models are present).
The descriptive Pareto frontier uses only selected runs and included repeats;
missing, invalid or varying parameter counts are omitted. Focus, hover or tap a
point for exact counts, mean error and observed replicate range. Backbone colours
stay consistent across selections and plots; marker shapes identify recorded variants.
The optional forecast-vs-actual replay adds one chart; error traces, heatmaps and
weight plots are not mounted on this page. Scores, diagnostic means and training/evaluation
settings are separate tabs of one numeric table. Diagnostic means use the same
included repeats as the selected score, with missing-value coverage shown on hover.

Choose validation, final test or explicitly labeled quick checks, then the dataset
and split plan, context window and horizon. The initial group prefers multiple
compatible runs over the newest singleton. Model selection persists across cells;
missing cells never become zero scores. Each cell matches unique seed/fold/split-size
keys independently. The top-bar picker supports search and up to six selected runs.

Completed TSLib15 archives can be registered as read-only comparison sources:

```sh
python scripts/register_comparison_archive.py results/tslib15-20260915 \
  --dataset data/external/etth1-1d16c8f/ETTh1.csv --workspace results/playground
```

Registration creates a source manifest under `comparison-archives`, not a training
job. `/api/v1/comparison-archives` verifies the complete audited model/arm/seed
matrix, dataset hash, shared forecast targets/origins, and metrics recomputed from
saved forecasts. Each model/arm retains its individual seed rows. Final-test scores,
original target units, exact split boundaries and the study task stay explicit.
The saved `.npz` files supply the optional overlay; dates come from the verified
source CSV. No model execution or retraining is involved. Unsupported diagnostics
remain unavailable. The displayed seed ranges are not the study report's bootstrap
intervals. A registered archive opens Compare on Final test; variant filtering in
the run picker exposes all 15 backbones without changing the current selection.

Run `scripts/check_compare.py` against the local server for browser coverage of the
registered study, selection persistence, numeric exports, graph count and mobile
layout. It blocks all job writes.
