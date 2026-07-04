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
  --iterations 120 \
  --batch-size 16 \
  --n-repeats 3 \
  --delay 0.04 \
  --seed 22
```

`--n-iterations` is accepted as an alias for `--iterations`. Increase
`--delay` for a slower presentation run, or set `--delay 0` for a fast
batch-style run that still writes the final visualization artifacts.

For long runs, contour previews are refreshed every `preview_every` sweeps by
default. This avoids the slowdown caused by recomputing a dense display-only
contour on every iteration as the campaign grows.

Each visualizer sweep runs `batch_size` simulations. With the default
`batch_size = 16` and `n_repeats = 3`, the proposal engine chooses 13 new
coordinates and 3 repeat coordinates. Set `n_new` and `n_repeats` in
`explore.params` to pin that split, or leave them as `auto` to let
`propose_next_sweep.py` choose.

The campaign has no hard cap on completed data points: total completed runs are
`initial_points + iterations * batch_size`. For example, 120 sweeps at the
default settings gives 1,940 completed runs, while 600 sweeps would give 9,620.
For speed, repeat scoring preselects about 50 informative existing coordinates
once the dataset is larger than that; model fitting and new-point scoring still
use the completed observations.

The visualizer defaults to `posterior_samples = 0` in `explore.params` for fast
interactive animation. Increase `posterior_samples` and
`preview_posterior_samples` when uncertainty bands matter more than speed.
`preview_points` and `preview_grid_size` only affect the displayed contour; they
do not change which experiments are proposed.

Near a lower log-scale boundary such as `x = 1`, prefer more points per sweep
or a denser display preview before jumping straight to hundreds of sweeps. A
typical refinement run is the default `--iterations 120 --batch-size 16 --n-repeats 3`,
which spends most of the extra budget on new coordinates while keeping a few
noise-check repeats. The visualizer also sets `length_scale_x = 0.18` by default,
uses `contour_fit = adaptive-linear`, and adds explicit edge bracket probes to
reduce over-smoothing near log-scale boundaries.

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

## Stopping

Common stopping checks:

- `y_c_std` is small across the target `x` range.
- Repeated labels near the contour are statistically stable.
- New proposed points stop moving the inferred contour.
- The contour resolution is fine enough for the downstream paper or design use.

The visualizer can stop automatically before `iterations` is exhausted. It
compares consecutive learned contours on the preview grid and stops when RMS
movement, maximum pointwise movement, and edge-region movement stay below their
configured tolerances for `convergence_patience` checks after
`convergence_min_iterations` sweeps. `convergence_mode` can compare `y`,
inverse `x`, or `both` views of the contour.
