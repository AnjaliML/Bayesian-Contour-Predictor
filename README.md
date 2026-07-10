# Bayesian Contour Predictor

Sequential active learning for noisy binary transition contours in a 2-D
parameter space.

This repository is process-agnostic: the core workflow talks about `x`, `y`,
and `id`, where `id = 1` is the positive regime and `id = 0` is the negative
regime. Physical names such as `Rr` and `Oh` are mapped into `x` and `y` at the
CLI boundary.

## Quick Start

Run the generic example:

```bash
python3 propose_next_sweep.py examples/generic_seed.csv \
  --outfile Sweep-1.csv \
  --n-simulations 8 \
  --seed 11
```

Run the drop-injection example with physical columns:

```bash
python3 propose_next_sweep.py examples/drop_injection_seed.csv \
  --x-col Rr \
  --y-col Oh \
  --mode monotone-y \
  --x-scale log10 \
  --y-scale log10 \
  --x-min 1 \
  --x-max 100 \
  --y-min 0.001 \
  --y-max 0.1 \
  --outfile Sweep-1.csv \
  --n-simulations 8 \
  --seed 11
```

The output rows are proposed experiments with `id = -1`. Replace each `-1`
with the observed `0` or `1`, append the rows to the campaign data, and rerun
the script for the next sweep.

Run the drop-injection workflow with a live browser visualization:

```bash
python3 end-to-end-with-visualization-drop-injection.py
```

This creates a local `visualization_runs/` directory, opens an animated canvas
view, calls `propose_next_sweep.py` for each batch, and calls
`classify_drops.py` as the experiment runner.

The visualizer reads defaults from [explore.params](explore.params), including
the initial random seed, bounds, batch size, and log/linear axis choices. The
drop-injection defaults use a log-log parameter space:

```text
Rr: 1 to 100
Oh: 0.001 to 0.1
```

Set the run length with:

```bash
python3 end-to-end-with-visualization-drop-injection.py --iterations 20
```

For a real simulator, start with the rearmable batch template instead of the
visualizer. Copy `examples/rearm_campaign.sh`, provide an initial completed CSV,
and set `BATCH_RUNNER` to an adapter that consumes proposed `x,y` rows and writes
completed `id` values. `assess_contour.py` stores a generic contour, transformed
SSE movement, bracket coverage, and a `refining`, `stalled`, or `converged`
state without access to a theoretical answer.

The parameter comparison is reproducible with:

```bash
python3 benchmarks/benchmark_contours.py --output benchmark-results.csv
```

Known synthetic contours exist only in this benchmark and are used only after
each batch to score normalized transformed-space SSE.

## Tests

```bash
python3 -m unittest
```

More detail lives in [docs/](docs/).
