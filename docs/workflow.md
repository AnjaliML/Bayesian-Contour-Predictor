# Iterative Workflow

1. Choose a bounded 2-D domain for `x` and `y`.
2. Run an initial random or space-filling set of experiments.
3. Store completed results in CSV form:

```text
caseId,x,y,id
```

4. Run the proposal script:

```bash
python3 propose_next_sweep.py Sweep-0.csv \
  --outfile Sweep-1.csv \
  --n-simulations 8 \
  --x-min 0 \
  --x-max 1 \
  --y-min 0.1 \
  --y-max 1.0 \
  --seed 11
```

5. Run the proposed experiments. Each output row starts with `id = -1`.
6. Replace `-1` with the observed `0` or `1`.
7. Append the completed rows to the campaign dataset.
8. Repeat until the contour is resolved enough for the scientific or
   engineering goal.

## End-To-End Visualizer

For the drop-injection example, run:

```bash
python3 end-to-end-with-visualization-drop-injection.py
```

The visualizer starts from an initial space-filling design, calls
`propose_next_sweep.py` to request each next batch, calls `classify_drops.py` to
simulate the experiment labels, and updates a browser animation as the campaign
advances.

The visualizer does not need an external initial CSV. If no completed data file
exists, it generates an initial space-filling design using `seed` and
`initial_points` from [../explore.params](../explore.params).

Useful options:

```bash
python3 end-to-end-with-visualization-drop-injection.py \
  --iterations 20 \
  --batch-size 8 \
  --n-repeats 0 \
  --delay 0.04 \
  --seed 22
```

`--n-iterations` is accepted as an alias for `--iterations`. Increase
`--delay` for a slower presentation run, or set `--delay 0` for a fast
batch-style run that still writes the final visualization artifacts.

The tuned example refreshes on every completed batch so convergence checks are
meaningful. A mandatory final refresh includes the final experimental batch.

Each visualizer sweep runs `batch_size` simulations. The deterministic example
uses eight new coordinates and no repeats. For a stochastic simulator, reserve
roughly 10--20 percent of the batch for repeats near disputed labels.

The campaign has no hard cap on completed data points: total completed runs are
`initial_points + iterations * batch_size`. The tuned maximum is 184 runs
before an earlier convergence stop.
For speed, repeat scoring preselects about 50 informative existing coordinates
once the dataset is larger than that; model fitting and new-point scoring still
use the completed observations.

The visualizer defaults to `posterior_samples = 0` in `explore.params` for fast
interactive animation. Increase `posterior_samples` and
`preview_posterior_samples` when uncertainty bands matter more than speed.
`preview_points` and `preview_grid_size` only affect the displayed contour; they
do not change which experiments are proposed.

The tuned profile uses `length_scale_x = 0.18` and `contour_fit = local-linear`.
The acquisition reserves candidates at both x edges and up to half of each
batch for persistent bracket bisection before filling remaining slots across
transformed x strata. It also reserves one of the default eight slots for a
randomized scarcity probe in an under-tested transformed-x region, fanned
around the predicted transition. For a targeted coverage pass, temporarily
increase `scarcity_fraction`; setting it to `0` restores bracket-only behavior.

The generated CSV files, `state.json`, and `index.html` are written under
`visualization_runs/` by default.

## Multiple CSV Inputs

The script accepts one or more files:

```bash
python3 propose_next_sweep.py Sweep-0.csv Sweep-1_completed.csv \
  --outfile Sweep-2.csv \
  --n-simulations 8
```

This lets each sweep remain as its own file while fitting against all completed
data.

## Repeated Coordinates

Duplicate `x,y` pairs are meaningful. They are aggregated as Bernoulli trials:

```text
n = number of repeats
k = number of positive outcomes
p_hat = k / n
```

The script deliberately proposes repeats near the inferred contour when repeats
are enabled. For new coordinates, it also bisects observed `id = 1` / `id = 0`
brackets and stratifies selections across the `x` range so one dense region
does not consume the whole batch.

## Constrained X Coordinates

Use a comma-separated allow-list when the experiment runner cannot accept
arbitrary `x` values:

```bash
python3 propose_next_sweep.py Sweep-0.csv \
  --x-min 1 --x-max 16 \
  --x-scale log10 \
  --x-candidates 1,1.5,2,3,4,6,8,10,13,16 \
  --n-simulations 16
```

The proposal engine evaluates and scores the allowed coordinates directly;
it does not round a continuous proposal after acquisition. Repeats are also
restricted, so any `--n-repeats` budget requires completed observations at
allowed `x` values.

## Stopping

Common stopping checks:

- `y_c_std` is small across the target `x` range.
- Repeated labels near the contour are statistically stable.
- New proposed points stop moving the inferred contour.
- The contour resolution is fine enough for the downstream paper or design use.

The visualizer and generic assessor do not stop on movement alone. They require
small RMS, maximum, and edge movement plus narrow observed y brackets, bounded
gaps between bracketed x anchors, and brackets at both x-domain edges. A stable
line without those resolution gates is `stalled`, not converged.

## Rearmable External-Simulator Loop

Copy [../examples/rearm_campaign.sh](../examples/rearm_campaign.sh). Put an
initial `caseId,x,y,id` file at
`campaign/completed/Sweep-0_completed.csv`, then run:

```bash
BATCH_RUNNER=/path/to/run_batch.sh \
CAMPAIGN_DIR=campaign \
bash examples/rearm_campaign.sh
```

The adapter contract is:

```text
run_batch.sh INPUT_PROPOSALS.csv OUTPUT_COMPLETED.csv
```

The adapter owns scheduler and process-specific details. It must preserve
`caseId,x,y` and replace every `id = -1` with `0` or `1`. The loop is durable:
it keeps completed batches, contour snapshots, and `state.json`; on `stalled`
it exits so a human or agent can adjust parameters and rearm it.

`assess_contour.py` reports consecutive-contour SSE because that is useful for
auditing, but stops on RMSE so the threshold is invariant to contour-grid size.
It never imports a theoretical boundary.
