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

The script deliberately proposes repeats near the inferred contour, especially
where existing labels disagree or only one run exists at an important point.

## Stopping

Common stopping checks:

- `y_c_std` is small across the target `x` range.
- Repeated labels near the contour are statistically stable.
- New proposed points stop moving the inferred contour.
- The contour resolution is fine enough for the downstream paper or design use.
