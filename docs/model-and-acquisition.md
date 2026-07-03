# Model And Acquisition

The proposal engine fits a lightweight probabilistic binary classifier from
completed rows and asks for new experiments that improve the estimate of:

```text
p(id = 1 | x, y) = 0.5
```

The implementation is dependency-free and deterministic given `--seed`.

## Generic Mode

`--mode generic` uses kernel-smoothed Bernoulli observations. Repeated
coordinates contribute `n` trials and `k` positive outcomes. The model uses a
Beta prior:

```text
p = (alpha + weighted positives) / (alpha + beta + weighted trials)
```

For each candidate `x`, the script scans `y` values and chooses the point where
the predicted probability is closest to `0.5`.

## Monotone-Y Mode

`--mode monotone-y` is for domains where the probability of `id = 1` changes
monotonically as `y` increases. The default direction is decreasing:

```text
p(id = 1 | x, y) =
  eps + (1 - 2 eps) sigmoid((y_c(x) - y) / w)
```

For positive `y` values, use:

```bash
--y-scale log10
```

This matches the drop-injection use case where increasing `Oh` suppresses
droplets.

For domains where the first coordinate spans orders of magnitude, use:

```bash
--x-scale log10
```

This makes kernel distances, sampled gaps, candidate coordinates, and novelty
scores operate in log space. In the drop-injection map this gives the low-`Rr`
region near `Rr = 1` meaningful resolution instead of letting the high end of a
linear `Rr` interval dominate the acquisition geometry.

Use `--monotone-direction increasing` when the positive regime becomes more
likely as `y` increases.

## Uncertainty

The script samples aggregate point rates from their Beta posteriors and refits
the contour for each sampled draw. It reports:

- `y_c_pred`: fitted contour location at the proposed `x`.
- `y_c_q05`: 5 percent posterior contour quantile.
- `y_c_q95`: 95 percent posterior contour quantile.
- `y_c_std`: posterior standard deviation of the contour location.

## Proposal Types

Each batch mixes:

- `new`: new coordinates near the current contour, weighted toward uncertain
  and sparsely sampled regions.
- `repeat`: existing coordinates where repeats help resolve noisy labels.

For an 8-run batch, the default is 5 new coordinates and 3 repeats. Override
this with:

```bash
--n-new 6 --n-repeats 2
```

## Scores And Reasons

The `score` ranks proposals using:

- closeness to predicted `p = 0.5`,
- local contour uncertainty,
- gaps in sampled `x`,
- distance from existing coordinates for new points,
- contradictory or low-repeat data for repeats.

Every row includes a `reason` column so the next experiment is explainable.
