# Drop Injection Mapping

The drop-injection example uses physical coordinates:

```text
Rr,Oh,id
```

For the process-agnostic workflow, map them as:

```text
Rr -> x
Oh -> y
```

The labels remain project-specific:

```text
id = 1 -> drops
id = 0 -> no drops
```

Run a proposal sweep with:

```bash
python3 propose_next_sweep.py examples/drop_injection_seed.csv \
  --x-col Rr \
  --y-col Oh \
  --mode monotone-y \
  --monotone-direction decreasing \
  --y-scale log10 \
  --y-min 0.01 \
  --y-max 0.2 \
  --outfile Sweep-1.csv \
  --n-simulations 8 \
  --seed 11
```

The output uses canonical `x,y` names. This keeps the learning method portable
across projects. In this physical case, interpret output `x` as `Rr` and output
`y` as `Oh`.

The older `classify_drops.py` helper evaluates one fixed hand-coded transition:

```text
Oh_c(Rr) = 0.0326 - 0.0398 exp(-0.348 Rr)
```

The new proposal engine is different: it learns a contour from campaign data
and proposes the next informative experiments.

## Visual End-To-End Demo

Run:

```bash
python3 end-to-end-with-visualization-drop-injection.py
```

By default this runs 60 active-learning sweeps. Use `--iterations N` or
`--n-iterations N` for longer campaigns.

This script keeps the drop-injection physical labels visible in the animation:

- `Rr` is shown on the horizontal axis.
- `Oh` is shown on a log-scaled vertical axis.
- green points are `id = 1`, drops.
- red points are `id = 0`, no drops.
- outlined points are the next proposed experiments.

Under the hood, acquisition still flows through the generic `x,y,id` proposal
engine, and each simulated experiment label is produced by `classify_drops.py`.
