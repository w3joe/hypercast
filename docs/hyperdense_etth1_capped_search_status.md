# HyperDense ETTh1 capped search

The user instructed “ok anyways proceed with the experiment then” after the proposed $100 staged allocation. This launches that $100 allocation, with at most two L4s, not the previously quoted $547 whole-stage reservation. The expected $60–$80 figure extrapolates three-epoch measurements and is not a guarantee of completing all fits.

The experiment retains the frozen training implementation and 224-fit scope: 12 equal settings across eight arms and two backbones at seed 2301 (192 fits), followed by two settings per backbone/arm at seed 2302 (32 fits). Jobs are reordered into 14 balanced 16-fit blocks. No new block starts unless all 16 reservations fit with $10 remaining. No accuracy-based pruning or unequal extra tuning is introduced. A failure stops subsequent waves; no automatic retries. A budget stop may leave the study incomplete, which must be reported explicitly.

New accounting for this allocation only: reserve `(hard timeout + 420 seconds) × hourly resource rate / 3600 × 1.25 + $0.05` per fit before submission. Once a successful job is archived and its app has stopped, settle using the conservative wall interval from submission through app shutdown, retaining the 420-second overhead and 25% margin. Original reservations remain recorded. Failed or unsubmitted attempts retain their full reservations. The hourly rate includes the L4, two CPUs and 4 GiB of memory. Earlier ledgers and frozen sources remain unchanged.

The new budget-controller and existing search tests passed (four tests); the source package was frozen after those checks. It contains 78 source/artifact files. The training/calibration controls previously passed 17 tests, and 16 corrected L4 calibration fits passed archive replay and initialization audits.

Live status is in `results/hyperdense-etth1-v2/search-001/ledger.json`; the durable driver log is `results/hyperdense-etth1-v2/search-001-driver.log`. The launch receipt records the process and command in `search-001-launch.json`. The frozen plan/data/source package is `search-prepared-v2-capped`. Returned checkpoints and forecasts are archived per job; cumulative development results are saved after every completed two-job wave.

The driver continues independently of the conversation until completion, a failure or a budget stop. Final-test data and confirmation/inference jobs are not supplied or authorized in this allocation. No periodic model monitor was enabled or reconfigured.
