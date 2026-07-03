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
  --y-scale log10 \
  --y-min 0.01 \
  --y-max 0.2 \
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

## Tests

```bash
python3 -m unittest
```

More detail lives in [docs/](docs/).
