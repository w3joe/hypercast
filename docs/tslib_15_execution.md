# Authorized all-15 execution

The user instructed “Proceed” after the $80, ten-L4, seven-hour plan. Start:
15 September 2026, 08:59:32 UTC. Report deadline: 15:59:32 UTC (16:59 London).
GPU finish target: 15:29:32 UTC. New allocation: `results/tslib15-20260915`.

No earlier allocation is resumed. Source packages are copied to separate frozen
directories; unrelated working-tree changes and historical experiment archives
are preserved. Eight focused tests, all 120 CPU model/arm gates, and synthetic
calibration/development/test worker paths passed before the first launch.

## CUDA repair

TimesNet's first CUDA identical-copy Adam check failed at maximum absolute
parameter discrepancy 1.1474e-5. Forward/gradient checks had passed. The first
calibration controller stopped submissions and collected active jobs. It
preserved 72 completed short fits and the failure. The second package enables
deterministic cuDNN convolution and disables convolution benchmarking for both
diagnostics and training. Tolerances remain atol 2e-6, rtol 2e-4. TimesNet passed
the repeated L4 gate and all eight three-epoch fits. All backbones are recalibrated
under the repaired protocol; first-pass data are superseded, not pooled.

## Prospective runtime amendment

Declared during label-free calibration, before tuning or held-out scoring:
retain all 15 models, all eight arms, three paired seeds, and patience 20. Try
the original two-rate/150-epoch design first, then the planned fixed-rate
150-epoch fallback. If that does not fit the remaining time, select a **uniform
125- or 100-epoch ceiling** solely by the calibrated ten-worker schedule. Do not
go below 100, drop arms/models, or select on accuracy. Report all ceiling hits.
This modifies the original fixed 150-epoch ceiling to prioritize complete,
balanced coverage inside the user's absolute deadline. It does not promise
optimization convergence.

The prospective fit allowance is `max measured epoch × ceiling × 1.3 + 60 seconds`,
rounded upward to 30 seconds, plus 30 seconds per fit in queue simulation. The
worker reserves the final 30 seconds of its fit allowance for returning state;
the global deadline retains a 90-second artifact margin. Warm containers remain
available for 60 seconds between calls. Full 150-epoch tuning is admitted only
if both stages fit; the uniform ceiling amendment applies to fixed-rate fallback.

This decision uses training/input-forward/checkpoint timings, not development or
test labels. The exact selected scope is recorded in `admission.json` before
the final-test bundle is created.

## Resource accounting

Verified account rate: L4 $0.80/hour + two physical CPU cores at $0.0473/hour each
+ 4 GiB at $0.008/GiB/hour = $0.9266/worker-hour. Workers have hard CPU/memory
limits and one L4, with at most ten active calls/containers. The ledger conservatively
charges ten workers for each entire controller/app session, including idle and
startup time, adds 10%, and retains $5 for ancillary costs. This deliberately
overstates actual active-worker time. Failed attempts remain represented.

At 6.5 hours the ten-worker envelope plus 10% and $5 is approximately $71.25;
the cap remains $80. The controller stops admission at $73 of this upper accounting
and admits no job whose timeout plus a return allowance reaches past the GPU
finish target. Provider billing may lag; ledger accounting is not a final invoice.

## Final label-free admission refinement

All 120 corrected calibrations completed at about 09:26 UTC. The 150-epoch
fixed-rate matrix projected to 9.08 hours; 125 epochs to 7.73 hours; 100 epochs
to 6.40 hours using the initial 30% timing margin. These figures assume every
fit reaches its ceiling and include per-fit overhead; they do not rely on early
stopping or any accuracy rankings.

For the final admission, use **25% above the maximum measured epoch** plus the
same 60-second per-fit and 30-second queue allowances. The 100-epoch matrix then
projects to **6.20 hours**. The GPU finish target moves to **15:49:32 UTC**, retaining
the absolute **15:59:32 UTC report deadline** and a ten-minute final reporting
reserve. Auditing is streamed during collection and the reporting code is already
implemented and tested. The controller additionally reserves five minutes in its
schedule admission. Neither the $80 cap nor ten-L4 concurrency changes.

This is a prospective operational revision using calibration timings only. The
test bundle remains absent at this decision. The fixed-rate comparison uses
learning rate 0.001, a uniform 100-epoch maximum and patience 20, retaining all
360 evaluation fits. The 25% allowance is a forecast margin, not a guarantee of
cloud runtime or convergence; deadline-interrupted fits must remain explicit.

FiLM's eight-arm calibration archive exceeded the 4 GiB memory limit. That
attempt is preserved as failed. The successful repair ran its full GPU gate
and native arm first, then seven separate arm jobs, and wrote archives to disk.
Memory remains capped at 4 GiB; model/training numerics are unchanged. The final
120-fit calibration set combines the 112 successful non-FiLM fits with those
eight repaired FiLM fits.
