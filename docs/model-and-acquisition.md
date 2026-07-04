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

`--contour-fit` controls the local contour model used in monotone-y mode:

- `local-constant`: fit one local `y_c` value at each candidate `x`.
- `local-linear`: fit a local line in transformed `x`/`y` space.
- `adaptive-linear`: use local-linear fitting near `x` domain edges and
  local-constant fitting elsewhere.

The default is `adaptive-linear`, which keeps the process-agnostic model but
reduces one-sided boundary bias without paying the full local-linear cost at
every interior candidate.

When completed data at an exact `x` already bracket the transition with both
`id = 1` and `id = 0`, tight brackets constrain the fitted contour at that
same `x`. This is still a monotone binary-contour rule, not a rule tied to any
particular physical process.

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

For monotone contours, new-point acquisition now uses both `y` from sampled `x`
locations and inverse `x` from sampled `y` levels. The inverse locator helps
probe boundary regions where the current `Y(x)` contour sits too high or too low
and the next useful experiment is a vertical bracket rather than another point
on the current line.

The monotone acquisition also proposes bracket refinements:

- exact bracket bisection when the same `x` has observed labels on both sides,
- local bracket bisection from nearby `x` neighborhoods,
- edge bracket expansion when a domain edge has only one observed label so far.

Final new-coordinate selection is stratified across transformed `x` bins, so a
dense or high-scoring region cannot consume the entire batch.
