# Contour benchmark

The benchmark deliberately separates acquisition from evaluation:

1. A frozen boundary function generates only binary `x,y,id` observations.
2. `propose_next_batch()` receives those observations and no boundary callback.
3. After each completed batch, the benchmark evaluates the learned contour on
   a fixed transformed-x grid.

The primary error is normalized transformed-space SSE:

```text
sum(((T_y(y_pred) - T_y(y_true)) / (T_y(y_max) - T_y(y_min))) ** 2)
```

RMSE is also reported because an SSE threshold changes when `--eval-points`
changes. Truth is never used as a real-campaign stopping signal.

## Directional profile comparison

A one-seed, five-family, deterministic screening run used 24 initial points,
five batches of eight, and 81 evaluation points. Before the acquisition fixes
in this branch, the screening results were:

| profile | mean final SSE | mean edge SSE | total proposal time |
| --- | ---: | ---: | ---: |
| prior `explore.params` | 0.1942 | 0.1742 | 26.52 s |
| balanced | 0.08791 | 0.06450 | 4.37 s |
| fast | 0.1714 | 0.1426 | 0.73 s |
| local-linear | 0.05961 | 0.05604 | 2.30 s |

After hard edge/bracket reservations, posterior fixes, and skipping repeat
scoring when `n_repeats = 0`, a local rerun of the selected profile produced
median final SSE `0.02783` across the same five families. On the held-out
built-in family, local-linear produced SSE `0.02783` in `3.15 s`; the prior
profile produced SSE `0.02879` in `20.66 s`. The prior profile had lower
edge-only SSE on that family, but local-linear matched total precision about
`6.5x` faster. These are directional single-seed timings, not universal
performance claims.

The selected starting profile is therefore:

```text
contour_fit = local-linear
grid_size = 21
posterior_samples = 0
transition_width = 0.04
label_noise = 0.005
length_scale_x = 0.18
```

Use the `balanced` profile when approximate uncertainty during acquisition is
worth the additional CPU time. For stochastic simulators, include nonzero
noise rates and repeats rather than reusing deterministic settings.

## Reproduce

Quick comparison:

```bash
python3 benchmarks/benchmark_contours.py \
  --profiles fast,local-linear \
  --output benchmark-results.csv
```

More reliable multi-seed/noise run:

```bash
python3 benchmarks/benchmark_contours.py \
  --seeds 11,29,47,71,101 \
  --initial-points 24 \
  --batch-size 8 \
  --iterations 12 \
  --eval-points 161 \
  --noise-rates 0,0.01,0.05 \
  --output benchmark-results.csv
```

Treat the repository's binary and size-based boundaries as one family: the
second is only a constant rescaling of the first, so it is not an independent
validation case.
