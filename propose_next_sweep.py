#!/usr/bin/env python3
"""Propose sequential experiments for a noisy 2-D binary transition contour."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


PROPOSAL_COLUMNS = [
    "caseId",
    "x",
    "y",
    "id",
    "proposal_type",
    "score",
    "p_positive_pred",
    "y_c_pred",
    "y_c_q05",
    "y_c_q95",
    "y_c_std",
    "n_existing",
    "k_existing",
    "reason",
]
LEGACY_COLUMNS = ["caseId", "x", "y", "id"]


@dataclass(frozen=True)
class Observation:
    case_id: str
    x: float
    y: float
    label: int


@dataclass(frozen=True)
class AggregatePoint:
    x: float
    y: float
    n: int
    k: int

    @property
    def rate(self) -> float:
        return self.k / self.n


@dataclass(frozen=True)
class Domain:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def x_span(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_span(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class ModelConfig:
    mode: str
    monotone_direction: str
    x_scale: str
    y_scale: str
    transition_width: float
    label_noise: float
    length_scale_x: float
    length_scale_y: float
    prior_alpha: float
    prior_beta: float
    grid_size: int
    posterior_samples: int


@dataclass(frozen=True)
class ContourEstimate:
    y_c_pred: float
    y_c_q05: float
    y_c_q95: float
    y_c_std: float


@dataclass(frozen=True)
class Proposal:
    x: float
    y: float
    proposal_type: str
    score: float
    p_positive_pred: float
    contour: ContourEstimate
    n_existing: int
    k_existing: int
    reason: str


def read_csv_files(
    paths: Sequence[Path],
    *,
    case_col: str,
    x_col: str,
    y_col: str,
    label_col: str,
) -> tuple[list[Observation], list[str]]:
    observations: list[Observation] = []
    all_case_ids: list[str] = []

    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [
                col
                for col in (case_col, x_col, y_col, label_col)
                if col not in (reader.fieldnames or [])
            ]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"{path} is missing required column(s): {joined}")

            for line_number, row in enumerate(reader, start=2):
                case_id = str(row[case_col]).strip()
                all_case_ids.append(case_id)
                raw_label = str(row[label_col]).strip()
                if raw_label == "-1":
                    continue
                if raw_label not in {"0", "1"}:
                    raise ValueError(
                        f"{path}:{line_number} has label {raw_label!r}; expected 0, 1, or -1"
                    )
                observations.append(
                    Observation(
                        case_id=case_id,
                        x=parse_float(row[x_col], path, line_number, x_col),
                        y=parse_float(row[y_col], path, line_number, y_col),
                        label=int(raw_label),
                    )
                )

    if not observations:
        raise ValueError("No completed rows with id 0 or 1 were found.")
    return observations, all_case_ids


def parse_float(value: str, path: Path, line_number: int, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number} column {column!r} has non-numeric value {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{line_number} column {column!r} is not finite")
    return parsed


def aggregate_observations(observations: Iterable[Observation]) -> list[AggregatePoint]:
    counts: dict[tuple[float, float], list[int]] = {}
    for obs in observations:
        bucket = counts.setdefault((obs.x, obs.y), [0, 0])
        bucket[0] += 1
        bucket[1] += obs.label
    return [
        AggregatePoint(x=x, y=y, n=n, k=k)
        for (x, y), (n, k) in sorted(counts.items())
    ]


def infer_domain(
    observations: Sequence[Observation],
    *,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> Domain:
    xs = [obs.x for obs in observations]
    ys = [obs.y for obs in observations]
    inferred = Domain(
        x_min=min(xs) if x_min is None else x_min,
        x_max=max(xs) if x_max is None else x_max,
        y_min=min(ys) if y_min is None else y_min,
        y_max=max(ys) if y_max is None else y_max,
    )
    return pad_degenerate_domain(inferred)


def pad_degenerate_domain(domain: Domain) -> Domain:
    x_min, x_max, y_min, y_max = domain.x_min, domain.x_max, domain.y_min, domain.y_max
    if x_min >= x_max:
        pad = abs(x_min) * 0.05 or 1.0
        x_min -= pad
        x_max += pad
    if y_min >= y_max:
        pad = abs(y_min) * 0.05 or 1.0
        y_min -= pad
        y_max += pad
    return Domain(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def default_length_scale(span: float) -> float:
    return max(span * 0.20, 1e-12)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def linspace(lower: float, upper: float, count: int) -> list[float]:
    if count <= 1:
        return [(lower + upper) / 2.0]
    step = (upper - lower) / (count - 1)
    return [lower + step * i for i in range(count)]


def minimize_bounded(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    iterations: int = 18,
) -> float:
    if lower >= upper:
        return lower
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0
    left = lower
    right = upper
    h = right - left
    c = left + inv_phi_sq * h
    d = left + inv_phi * h
    fc = objective(c)
    fd = objective(d)
    for _ in range(iterations):
        if fc < fd:
            right = d
            d = c
            fd = fc
            h = inv_phi * h
            c = left + inv_phi_sq * h
            fc = objective(c)
        else:
            left = c
            c = d
            fc = fd
            h = inv_phi * h
            d = left + inv_phi * h
            fd = objective(d)
    return (left + right) / 2.0


def transformed_value(value: float, scale: str, axis_name: str) -> float:
    if scale == "linear":
        return value
    if value <= 0:
        raise ValueError(
            f"--{axis_name}-scale log10 requires all {axis_name} values and bounds to be positive"
        )
    return math.log10(value)


def inverse_transformed_value(value: float, scale: str) -> float:
    if scale == "linear":
        return value
    return 10**value


def transformed_x(value: float, config: ModelConfig) -> float:
    return transformed_value(value, config.x_scale, "x")


def transformed_y(value: float, scale: str) -> float:
    return transformed_value(value, scale, "y")


def inverse_transformed_y(value: float, scale: str) -> float:
    return inverse_transformed_value(value, scale)


def transformed_span(lower: float, upper: float, scale: str, axis_name: str) -> float:
    return transformed_value(upper, scale, axis_name) - transformed_value(
        lower, scale, axis_name
    )


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normal_weight(distance: float, length_scale: float) -> float:
    scaled = distance / max(length_scale, 1e-12)
    return math.exp(-0.5 * scaled * scaled)


def kernel_predict(
    x: float,
    y: float,
    aggregates: Sequence[AggregatePoint],
    config: ModelConfig,
    *,
    sampled_rates: dict[tuple[float, float], float] | None = None,
) -> tuple[float, float]:
    weighted_k = 0.0
    weighted_n = 0.0
    for point in aggregates:
        weight_x = normal_weight(
            transformed_x(x, config) - transformed_x(point.x, config),
            config.length_scale_x,
        )
        weight_y = normal_weight(
            transformed_y(y, config.y_scale) - transformed_y(point.y, config.y_scale),
            config.length_scale_y,
        )
        weight = weight_x * weight_y
        rate = sampled_rates[(point.x, point.y)] if sampled_rates else point.rate
        weighted_k += weight * point.n * rate
        weighted_n += weight * point.n

    numerator = config.prior_alpha + weighted_k
    denominator = config.prior_alpha + config.prior_beta + weighted_n
    probability = numerator / denominator
    posterior_std = math.sqrt(
        max(probability * (1.0 - probability), 0.0) / max(denominator + 1.0, 1e-12)
    )
    return probability, posterior_std


def monotone_probability(
    y: float,
    y_c: float,
    *,
    direction: str,
    y_scale: str,
    transition_width: float,
    label_noise: float,
) -> float:
    y_t = transformed_y(y, y_scale)
    y_c_t = transformed_y(y_c, y_scale)
    if direction == "decreasing":
        margin = (y_c_t - y_t) / transition_width
    else:
        margin = (y_t - y_c_t) / transition_width
    return label_noise + (1.0 - 2.0 * label_noise) * logistic(margin)


def monotone_negative_log_likelihood(
    x: float,
    y_c: float,
    aggregates: Sequence[AggregatePoint],
    config: ModelConfig,
    *,
    sampled_rates: dict[tuple[float, float], float] | None = None,
) -> float:
    loss = 0.0
    for point in aggregates:
        weight = normal_weight(
            transformed_x(x, config) - transformed_x(point.x, config),
            config.length_scale_x,
        )
        if weight < 1e-9:
            continue
        rate = sampled_rates[(point.x, point.y)] if sampled_rates else point.rate
        k = point.n * rate
        p = clamp(
            monotone_probability(
                point.y,
                y_c,
                direction=config.monotone_direction,
                y_scale=config.y_scale,
                transition_width=config.transition_width,
                label_noise=config.label_noise,
            ),
            1e-9,
            1.0 - 1e-9,
        )
        loss -= weight * (k * math.log(p) + (point.n - k) * math.log(1.0 - p))
    return loss


def weighted_points_for_x(
    x: float,
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
) -> list[tuple[AggregatePoint, float]]:
    x_t = transformed_x(x, config)
    length_scale_x = boundary_adjusted_length_scale_x(x, domain, config)
    weighted: list[tuple[AggregatePoint, float]] = []
    for point in aggregates:
        weight = normal_weight(
            x_t - transformed_x(point.x, config),
            length_scale_x,
        )
        if weight >= 1e-9:
            weighted.append((point, weight))
    return weighted


def monotone_negative_log_likelihood_weighted(
    y_c: float,
    weighted_points: Sequence[tuple[AggregatePoint, float]],
    config: ModelConfig,
    *,
    sampled_rates: dict[tuple[float, float], float] | None = None,
) -> float:
    loss = 0.0
    for point, weight in weighted_points:
        rate = sampled_rates[(point.x, point.y)] if sampled_rates else point.rate
        k = point.n * rate
        p = clamp(
            monotone_probability(
                point.y,
                y_c,
                direction=config.monotone_direction,
                y_scale=config.y_scale,
                transition_width=config.transition_width,
                label_noise=config.label_noise,
            ),
            1e-9,
            1.0 - 1e-9,
        )
        loss -= weight * (k * math.log(p) + (point.n - k) * math.log(1.0 - p))
    return loss


def estimate_monotone_y_c(
    x: float,
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
    *,
    sampled_rates: dict[tuple[float, float], float] | None = None,
) -> float:
    lower = transformed_y(domain.y_min, config.y_scale)
    upper = transformed_y(domain.y_max, config.y_scale)
    candidates_t = linspace(lower, upper, max(config.grid_size, 5))
    weighted_points = weighted_points_for_x(x, aggregates, domain, config)
    losses = [
        monotone_negative_log_likelihood_weighted(
            inverse_transformed_y(value, config.y_scale),
            weighted_points,
            config,
            sampled_rates=sampled_rates,
        )
        for value in candidates_t
    ]
    best_index = min(range(len(candidates_t)), key=lambda index: losses[index])
    bracket_lower = candidates_t[max(0, best_index - 1)]
    bracket_upper = candidates_t[min(len(candidates_t) - 1, best_index + 1)]
    best_t = minimize_bounded(
        lambda value: monotone_negative_log_likelihood_weighted(
            inverse_transformed_y(value, config.y_scale),
            weighted_points,
            config,
            sampled_rates=sampled_rates,
        ),
        bracket_lower,
        bracket_upper,
    )
    return inverse_transformed_y(best_t, config.y_scale)


def estimate_generic_y_c(
    x: float,
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
    *,
    sampled_rates: dict[tuple[float, float], float] | None = None,
) -> float:
    lower = transformed_y(domain.y_min, config.y_scale)
    upper = transformed_y(domain.y_max, config.y_scale)
    y_values = [
        inverse_transformed_y(value, config.y_scale)
        for value in linspace(lower, upper, max(config.grid_size, 5))
    ]
    return min(
        y_values,
        key=lambda y: abs(
            kernel_predict(x, y, aggregates, config, sampled_rates=sampled_rates)[0] - 0.5
        ),
    )


def estimate_y_c(
    x: float,
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
    *,
    sampled_rates: dict[tuple[float, float], float] | None = None,
) -> float:
    if config.mode == "monotone-y":
        return estimate_monotone_y_c(
            x, aggregates, domain, config, sampled_rates=sampled_rates
        )
    return estimate_generic_y_c(x, aggregates, domain, config, sampled_rates=sampled_rates)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty sample.")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sample_rates(
    aggregates: Sequence[AggregatePoint], rng: random.Random
) -> dict[tuple[float, float], float]:
    return {
        (point.x, point.y): rng.betavariate(point.k + 1.0, point.n - point.k + 1.0)
        for point in aggregates
    }


def contour_estimate(
    x: float,
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
    rng: random.Random,
) -> ContourEstimate:
    predicted = estimate_y_c(x, aggregates, domain, config)
    samples = [
        estimate_y_c(
            x,
            aggregates,
            domain,
            config,
            sampled_rates=sample_rates(aggregates, rng),
        )
        for _ in range(max(config.posterior_samples, 0))
    ]
    if len(samples) >= 2:
        std = statistics.pstdev(samples)
        q05 = quantile(samples, 0.05)
        q95 = quantile(samples, 0.95)
    elif len(samples) == 1:
        std = 0.0
        q05 = samples[0]
        q95 = samples[0]
    else:
        std = 0.0
        q05 = predicted
        q95 = predicted
    return ContourEstimate(
        y_c_pred=predicted,
        y_c_q05=q05,
        y_c_q95=q95,
        y_c_std=std,
    )


def predict_probability(
    x: float,
    y: float,
    contour: ContourEstimate,
    aggregates: Sequence[AggregatePoint],
    config: ModelConfig,
) -> tuple[float, float]:
    if config.mode == "monotone-y":
        p = monotone_probability(
            y,
            contour.y_c_pred,
            direction=config.monotone_direction,
            y_scale=config.y_scale,
            transition_width=config.transition_width,
            label_noise=config.label_noise,
        )
        return p, 0.0
    return kernel_predict(x, y, aggregates, config)


def exact_existing_counts(
    x: float, y: float, aggregates_by_coord: dict[tuple[float, float], AggregatePoint]
) -> tuple[int, int]:
    point = aggregates_by_coord.get((x, y))
    if point is None:
        return 0, 0
    return point.n, point.k


def normalized_distance_to_existing(
    x: float,
    y: float,
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
) -> float:
    scale_x = transformed_span(domain.x_min, domain.x_max, config.x_scale, "x") or 1.0
    scale_y = transformed_span(domain.y_min, domain.y_max, config.y_scale, "y") or 1.0
    x_t = transformed_x(x, config)
    y_t = transformed_y(y, config.y_scale)
    return min(
        math.hypot(
            (x_t - transformed_x(point.x, config)) / scale_x,
            (y_t - transformed_y(point.y, config.y_scale)) / scale_y,
        )
        for point in aggregates
    )


def boundary_focus(x: float, domain: Domain, config: ModelConfig) -> float:
    lower = transformed_value(domain.x_min, config.x_scale, "x")
    upper = transformed_value(domain.x_max, config.x_scale, "x")
    span = max(upper - lower, 1e-12)
    x_t = transformed_x(x, config)
    edge_distance = min(abs(x_t - lower), abs(upper - x_t))
    return clamp(1.0 - edge_distance / (0.16 * span), 0.0, 1.0)


def boundary_adjusted_length_scale_x(
    x: float, domain: Domain, config: ModelConfig
) -> float:
    focus = boundary_focus(x, domain, config)
    adjusted = config.length_scale_x / (1.0 + 5.0 * focus)
    return max(adjusted, config.length_scale_x * 0.12, 1e-12)


def x_gap_score(
    x: float, aggregates: Sequence[AggregatePoint], domain: Domain, config: ModelConfig
) -> float:
    xs = sorted({point.x for point in aggregates})
    if not xs:
        return 1.0
    augmented = [
        transformed_value(value, config.x_scale, "x")
        for value in [domain.x_min, *xs, domain.x_max]
    ]
    x_t = transformed_x(x, config)
    containing_gap = 0.0
    for left, right in zip(augmented, augmented[1:]):
        if left <= x_t <= right:
            containing_gap = max(containing_gap, right - left)
    nearest_x_distance = min(
        abs(x_t - transformed_value(existing_x, config.x_scale, "x"))
        for existing_x in xs
    )
    x_span = transformed_span(domain.x_min, domain.x_max, config.x_scale, "x")
    return clamp(max(containing_gap, nearest_x_distance) / max(x_span, 1e-12), 0.0, 1.0)


def choose_batch_counts(
    n_simulations: int,
    n_new: int | None,
    n_repeats: int | None,
) -> tuple[int, int]:
    if n_new is not None and n_repeats is not None:
        if n_new + n_repeats != n_simulations:
            raise ValueError("--n-new plus --n-repeats must equal --n-simulations")
        return n_new, n_repeats
    if n_new is not None:
        if n_new > n_simulations:
            raise ValueError("--n-new cannot exceed --n-simulations")
        return n_new, n_simulations - n_new
    if n_repeats is not None:
        if n_repeats > n_simulations:
            raise ValueError("--n-repeats cannot exceed --n-simulations")
        return n_simulations - n_repeats, n_repeats
    repeats = min(3, max(1, round(n_simulations * 3 / 8))) if n_simulations > 1 else 0
    return n_simulations - repeats, repeats


def repeat_candidate_points(
    aggregates: Sequence[AggregatePoint], requested_count: int
) -> list[AggregatePoint]:
    if len(aggregates) <= 50:
        return list(aggregates)
    limit = max(50, requested_count * 16)

    def cheap_repeat_score(point: AggregatePoint) -> tuple[float, float, float]:
        disagreement = 1.0 - min(1.0, abs(point.rate - 0.5) * 2.0)
        low_repeat = 1.0 / max(point.n, 1)
        single_run = 1.0 if point.n == 1 else 0.0
        return (
            0.55 * disagreement + 0.35 * low_repeat + 0.10 * single_run,
            -point.n,
            point.x,
        )

    return sorted(aggregates, key=cheap_repeat_score, reverse=True)[:limit]


def propose_repeats(
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
    rng: random.Random,
    count: int,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    contour_cache: dict[float, ContourEstimate] = {}
    for point in repeat_candidate_points(aggregates, count):
        if point.x not in contour_cache:
            contour_cache[point.x] = contour_estimate(point.x, aggregates, domain, config, rng)
        contour = contour_cache[point.x]
        p_pred, p_std = predict_probability(point.x, point.y, contour, aggregates, config)
        near_contour = 1.0 - min(1.0, abs(p_pred - 0.5) * 2.0)
        disagreement = 1.0 - min(1.0, abs(point.rate - 0.5) * 2.0)
        low_repeat = 1.0 / max(point.n, 1)
        contour_uncertainty = contour_uncertainty_fraction(contour, domain, config)
        score = (
            0.35 * near_contour
            + 0.30 * disagreement
            + 0.20 * low_repeat
            + 0.10 * contour_uncertainty
            + 0.05 * p_std
        )
        reasons = []
        if 0 < point.k < point.n:
            reasons.append("existing repeats disagree")
        if near_contour >= 0.7:
            reasons.append("near predicted p=0.5 contour")
        if point.n == 1:
            reasons.append("only one run at an informative coordinate")
        if contour_uncertainty >= 0.05:
            reasons.append("local contour uncertainty remains high")
        if not reasons:
            reasons.append("repeat reduces label noise at an informative coordinate")
        proposals.append(
            Proposal(
                x=point.x,
                y=point.y,
                proposal_type="repeat",
                score=score,
                p_positive_pred=p_pred,
                contour=contour,
                n_existing=point.n,
                k_existing=point.k,
                reason="; ".join(reasons),
            )
        )

    proposals.sort(key=lambda proposal: (-proposal.score, proposal.x, proposal.y))
    return proposals[:count]


def candidate_x_values(
    domain: Domain, aggregates: Sequence[AggregatePoint], config: ModelConfig, count: int
) -> list[float]:
    lower = transformed_value(domain.x_min, config.x_scale, "x")
    upper = transformed_value(domain.x_max, config.x_scale, "x")
    base = [inverse_transformed_value(value, config.x_scale) for value in linspace(lower, upper, count)]
    xs = sorted({point.x for point in aggregates})
    midpoints = [
        inverse_transformed_value(
            (
                transformed_value(left, config.x_scale, "x")
                + transformed_value(right, config.x_scale, "x")
            )
            / 2.0,
            config.x_scale,
        )
        for left, right in zip(xs, xs[1:])
    ]
    values = [*base, *midpoints]
    deduped: list[float] = []
    x_span = transformed_span(domain.x_min, domain.x_max, config.x_scale, "x")
    for value in sorted(values):
        if (
            not deduped
            or abs(
                transformed_value(value, config.x_scale, "x")
                - transformed_value(deduped[-1], config.x_scale, "x")
            )
            > x_span * 1e-9
        ):
            deduped.append(value)
    return deduped


def contour_uncertainty_fraction(
    contour: ContourEstimate, domain: Domain, config: ModelConfig
) -> float:
    lower = max(min(contour.y_c_q05, contour.y_c_q95), domain.y_min)
    upper = min(max(contour.y_c_q05, contour.y_c_q95), domain.y_max)
    if lower <= 0 and config.y_scale == "log10":
        return 0.0
    spread = abs(
        transformed_value(upper, config.y_scale, "y")
        - transformed_value(lower, config.y_scale, "y")
    )
    span = transformed_span(domain.y_min, domain.y_max, config.y_scale, "y")
    return clamp(spread / max(span, 1e-12), 0.0, 1.0)


def y_probe_values_near_contour(
    contour: ContourEstimate,
    domain: Domain,
    config: ModelConfig,
    *,
    edge_focus: float = 0.0,
) -> list[tuple[float, float]]:
    center_t = transformed_value(contour.y_c_pred, config.y_scale, "y")
    lower_t = transformed_value(domain.y_min, config.y_scale, "y")
    upper_t = transformed_value(domain.y_max, config.y_scale, "y")
    span_t = max(upper_t - lower_t, 1e-12)
    q05_t = transformed_value(
        clamp(contour.y_c_q05, domain.y_min, domain.y_max), config.y_scale, "y"
    )
    q95_t = transformed_value(
        clamp(contour.y_c_q95, domain.y_min, domain.y_max), config.y_scale, "y"
    )
    half_spread_t = max(abs(q95_t - q05_t) / 4.0, 0.025 * span_t)
    offsets = [0.0, -half_spread_t, half_spread_t]
    if edge_focus >= 0.25:
        offsets.extend(
            [
                -2.0 * half_spread_t,
                2.0 * half_spread_t,
                -4.0 * half_spread_t,
                4.0 * half_spread_t,
            ]
        )

    probes: list[tuple[float, float]] = []
    seen: set[float] = set()
    for offset in offsets:
        y_t = clamp(center_t + offset, lower_t, upper_t)
        if any(abs(y_t - existing) <= span_t * 1e-9 for existing in seen):
            continue
        seen.add(y_t)
        probes.append(
            (
                inverse_transformed_value(y_t, config.y_scale),
                clamp(abs(offset) / max(0.10 * span_t, 1e-12), 0.0, 1.0),
            )
        )
    return probes


def y_offsets_near_contour(
    contour: ContourEstimate, domain: Domain, config: ModelConfig
) -> list[float]:
    return [
        value
        for value, _ in y_probe_values_near_contour(contour, domain, config)
    ]


def candidate_y_values(
    domain: Domain,
    config: ModelConfig,
    contour_path: Sequence[tuple[float, ContourEstimate]],
    count: int,
) -> list[float]:
    lower = transformed_value(domain.y_min, config.y_scale, "y")
    upper = transformed_value(domain.y_max, config.y_scale, "y")
    span = max(upper - lower, 1e-12)
    contour_values = [
        transformed_value(contour.y_c_pred, config.y_scale, "y")
        for _, contour in contour_path
    ]
    if contour_values:
        lower = max(lower, min(contour_values) - 0.16 * span)
        upper = min(upper, max(contour_values) + 0.16 * span)
    return [
        inverse_transformed_value(value, config.y_scale)
        for value in linspace(lower, upper, max(count, 5))
    ]


def x_candidates_for_y_level(
    y: float,
    contour_path: Sequence[tuple[float, ContourEstimate]],
    domain: Domain,
    config: ModelConfig,
) -> list[float]:
    if not contour_path:
        return []
    y_t = transformed_value(y, config.y_scale, "y")
    candidates: list[float] = []
    ordered = sorted(contour_path, key=lambda item: item[0])
    for (left_x, left_contour), (right_x, right_contour) in zip(ordered, ordered[1:]):
        left_y = transformed_value(left_contour.y_c_pred, config.y_scale, "y")
        right_y = transformed_value(right_contour.y_c_pred, config.y_scale, "y")
        if abs(left_y - y_t) <= 1e-12:
            candidates.append(left_x)
            continue
        if abs(right_y - y_t) <= 1e-12:
            candidates.append(right_x)
            continue
        if (left_y - y_t) * (right_y - y_t) > 0:
            continue
        denominator = right_y - left_y
        if abs(denominator) <= 1e-12:
            continue
        fraction = clamp((y_t - left_y) / denominator, 0.0, 1.0)
        left_x_t = transformed_value(left_x, config.x_scale, "x")
        right_x_t = transformed_value(right_x, config.x_scale, "x")
        x_t = left_x_t + fraction * (right_x_t - left_x_t)
        candidates.append(
            inverse_transformed_value(
                clamp(
                    x_t,
                    transformed_value(domain.x_min, config.x_scale, "x"),
                    transformed_value(domain.x_max, config.x_scale, "x"),
                ),
                config.x_scale,
            )
        )

    if candidates:
        return sorted(set(candidates))

    nearest_x, _ = min(
        ordered,
        key=lambda item: abs(
            transformed_value(item[1].y_c_pred, config.y_scale, "y") - y_t
        ),
    )
    return [nearest_x]


def propose_new_points(
    aggregates: Sequence[AggregatePoint],
    domain: Domain,
    config: ModelConfig,
    rng: random.Random,
    count: int,
) -> list[Proposal]:
    if count <= 0:
        return []
    aggregates_by_coord = {(point.x, point.y): point for point in aggregates}
    proposals: list[Proposal] = []
    x_values = candidate_x_values(domain, aggregates, config, max(config.grid_size, count * 4))
    contour_cache: dict[float, ContourEstimate] = {}

    def contour_for(x: float) -> ContourEstimate:
        if x not in contour_cache:
            contour_cache[x] = contour_estimate(x, aggregates, domain, config, rng)
        return contour_cache[x]

    def add_candidate(
        x: float,
        y: float,
        *,
        source: str,
        vertical_probe: float,
    ) -> None:
        n_existing, _ = exact_existing_counts(x, y, aggregates_by_coord)
        if n_existing:
            return
        contour = contour_for(x)
        p_pred, p_std = predict_probability(x, y, contour, aggregates, config)
        near_contour = 1.0 - min(1.0, abs(p_pred - 0.5) * 2.0)
        novelty = clamp(
            normalized_distance_to_existing(x, y, aggregates, domain, config) / 0.20,
            0.0,
            1.0,
        )
        gap = x_gap_score(x, aggregates, domain, config)
        contour_uncertainty = contour_uncertainty_fraction(contour, domain, config)
        edge_focus = boundary_focus(x, domain, config)
        y_span = max(
            transformed_span(domain.y_min, domain.y_max, config.y_scale, "y"),
            1e-12,
        )
        vertical_gap = abs(
            transformed_value(y, config.y_scale, "y")
            - transformed_value(contour.y_c_pred, config.y_scale, "y")
        )
        adaptive_vertical_probe = max(
            vertical_probe,
            clamp(vertical_gap / (0.10 * y_span), 0.0, 1.0),
        )
        inverse_locator = 1.0 if source == "x-from-y locator" else 0.0
        score = (
            0.30 * near_contour
            + 0.22 * contour_uncertainty
            + 0.17 * gap
            + 0.13 * novelty
            + 0.05 * p_std
            + 0.08 * edge_focus
            + 0.22 * edge_focus * adaptive_vertical_probe
            + 0.08 * inverse_locator
        )
        reasons = ["new coordinate near predicted contour"]
        if source == "x-from-y locator":
            reasons.append("adaptive x-from-y locator")
        if gap >= 0.25:
            reasons.append("large gap in sampled x values")
        if contour_uncertainty >= 0.05:
            reasons.append("high local contour uncertainty")
        if novelty >= 0.75:
            reasons.append("away from existing coordinates")
        if edge_focus >= 0.5 and adaptive_vertical_probe >= 0.5:
            reasons.append("boundary bracket probe")
        proposals.append(
            Proposal(
                x=x,
                y=y,
                proposal_type="new",
                score=score,
                p_positive_pred=p_pred,
                contour=contour,
                n_existing=0,
                k_existing=0,
                reason="; ".join(reasons),
            )
        )

    for x in x_values:
        contour = contour_for(x)
        edge_focus = boundary_focus(x, domain, config)
        for y, vertical_probe in y_probe_values_near_contour(
            contour, domain, config, edge_focus=edge_focus
        ):
            add_candidate(x, y, source="y-from-x locator", vertical_probe=vertical_probe)

    contour_path = [(x, contour_for(x)) for x in x_values]
    for y in candidate_y_values(domain, config, contour_path, max(config.grid_size, count * 3)):
        for x in x_candidates_for_y_level(y, contour_path, domain, config):
            add_candidate(x, y, source="x-from-y locator", vertical_probe=0.0)

    proposals.sort(key=lambda proposal: (-proposal.score, proposal.x, proposal.y))
    selected: list[Proposal] = []
    for proposal in proposals:
        if len(selected) >= count:
            break
        if all(
            math.hypot(
                (
                    transformed_value(proposal.x, config.x_scale, "x")
                    - transformed_value(chosen.x, config.x_scale, "x")
                )
                / max(transformed_span(domain.x_min, domain.x_max, config.x_scale, "x"), 1e-12),
                (
                    transformed_value(proposal.y, config.y_scale, "y")
                    - transformed_value(chosen.y, config.y_scale, "y")
                )
                / max(transformed_span(domain.y_min, domain.y_max, config.y_scale, "y"), 1e-12),
            )
            >= 0.03
            for chosen in selected
        ):
            selected.append(proposal)
    return selected


def propose_next_batch(
    observations: Sequence[Observation],
    *,
    domain: Domain,
    config: ModelConfig,
    n_simulations: int,
    n_new: int | None,
    n_repeats: int | None,
    seed: int,
) -> list[Proposal]:
    if n_simulations <= 0:
        raise ValueError("--n-simulations must be positive")
    aggregates = aggregate_observations(observations)
    new_count, repeat_count = choose_batch_counts(n_simulations, n_new, n_repeats)
    rng = random.Random(seed)

    repeats = propose_repeats(aggregates, domain, config, rng, repeat_count)
    new_points = propose_new_points(aggregates, domain, config, rng, new_count)

    if len(repeats) < repeat_count:
        new_points.extend(
            propose_new_points(
                aggregates,
                domain,
                config,
                rng,
                new_count + repeat_count - len(repeats),
            )[len(new_points) :]
        )
    if len(new_points) < new_count:
        repeats.extend(
            propose_repeats(
                aggregates,
                domain,
                config,
                rng,
                repeat_count + new_count - len(new_points),
            )[len(repeats) :]
        )

    batch = [*new_points[:new_count], *repeats[:repeat_count]]
    batch.sort(key=lambda proposal: (proposal.proposal_type != "new", -proposal.score))
    return batch[:n_simulations]


def next_case_id_generator(existing_case_ids: Sequence[str]) -> Iterable[str]:
    numeric_ids: list[int] = []
    for case_id in existing_case_ids:
        try:
            numeric_ids.append(int(case_id))
        except ValueError:
            continue
    if len(numeric_ids) == len(existing_case_ids):
        next_id = max(numeric_ids, default=0) + 1
        while True:
            yield str(next_id)
            next_id += 1
    index = 1
    existing = set(existing_case_ids)
    while True:
        candidate = f"proposal_{index:04d}"
        if candidate not in existing:
            yield candidate
        index += 1


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def proposals_to_rows(
    proposals: Sequence[Proposal],
    *,
    existing_case_ids: Sequence[str],
    legacy_columns_only: bool,
) -> list[dict[str, str]]:
    case_ids = next_case_id_generator(existing_case_ids)
    rows: list[dict[str, str]] = []
    for proposal in proposals:
        row = {
            "caseId": next(case_ids),
            "x": format_float(proposal.x),
            "y": format_float(proposal.y),
            "id": "-1",
            "proposal_type": proposal.proposal_type,
            "score": format_float(proposal.score),
            "p_positive_pred": format_float(proposal.p_positive_pred),
            "y_c_pred": format_float(proposal.contour.y_c_pred),
            "y_c_q05": format_float(proposal.contour.y_c_q05),
            "y_c_q95": format_float(proposal.contour.y_c_q95),
            "y_c_std": format_float(proposal.contour.y_c_std),
            "n_existing": str(proposal.n_existing),
            "k_existing": str(proposal.k_existing),
            "reason": proposal.reason,
        }
        rows.append({col: row[col] for col in (LEGACY_COLUMNS if legacy_columns_only else PROPOSAL_COLUMNS)})
    return rows


def write_rows(rows: Sequence[dict[str, str]], outfile: Path | None, *, legacy: bool) -> None:
    columns = LEGACY_COLUMNS if legacy else PROPOSAL_COLUMNS
    if outfile is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return
    with outfile.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Propose the next batch for noisy 2-D binary contour learning."
    )
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--n-simulations", type=int, default=8)
    parser.add_argument("--n-new", type=int)
    parser.add_argument("--n-repeats", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-col", default="caseId")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--label-col", default="id")
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--mode", choices=["generic", "monotone-y"], default="generic")
    parser.add_argument(
        "--monotone-direction",
        choices=["decreasing", "increasing"],
        default="decreasing",
        help="Direction for p(id=1) as y increases when --mode monotone-y is used.",
    )
    parser.add_argument("--x-scale", choices=["linear", "log10"], default="linear")
    parser.add_argument("--y-scale", choices=["linear", "log10"], default="linear")
    parser.add_argument("--transition-width", type=float, default=0.10)
    parser.add_argument("--label-noise", type=float, default=0.02)
    parser.add_argument("--length-scale-x", type=float)
    parser.add_argument("--length-scale-y", type=float)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--posterior-samples", type=int, default=80)
    parser.add_argument("--legacy-columns-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.transition_width <= 0:
        raise ValueError("--transition-width must be positive")
    if not 0 <= args.label_noise < 0.5:
        raise ValueError("--label-noise must be in [0, 0.5)")
    if args.prior_alpha <= 0 or args.prior_beta <= 0:
        raise ValueError("--prior-alpha and --prior-beta must be positive")
    if args.grid_size < 5:
        raise ValueError("--grid-size must be at least 5")
    if args.posterior_samples < 0:
        raise ValueError("--posterior-samples cannot be negative")


def validate_domain_scales(domain: Domain, args: argparse.Namespace) -> None:
    transformed_span(domain.x_min, domain.x_max, args.x_scale, "x")
    transformed_span(domain.y_min, domain.y_max, args.y_scale, "y")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        observations, all_case_ids = read_csv_files(
            args.csv_files,
            case_col=args.case_col,
            x_col=args.x_col,
            y_col=args.y_col,
            label_col=args.label_col,
        )
        domain = infer_domain(
            observations,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )
        validate_domain_scales(domain, args)
        config = ModelConfig(
            mode=args.mode,
            monotone_direction=args.monotone_direction,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            transition_width=args.transition_width,
            label_noise=args.label_noise,
            length_scale_x=args.length_scale_x
            or default_length_scale(
                transformed_span(domain.x_min, domain.x_max, args.x_scale, "x")
            ),
            length_scale_y=args.length_scale_y
            or default_length_scale(
                transformed_span(domain.y_min, domain.y_max, args.y_scale, "y")
            ),
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
            grid_size=args.grid_size,
            posterior_samples=args.posterior_samples,
        )
        proposals = propose_next_batch(
            observations,
            domain=domain,
            config=config,
            n_simulations=args.n_simulations,
            n_new=args.n_new,
            n_repeats=args.n_repeats,
            seed=args.seed,
        )
        rows = proposals_to_rows(
            proposals,
            existing_case_ids=all_case_ids,
            legacy_columns_only=args.legacy_columns_only,
        )
        write_rows(rows, args.outfile, legacy=args.legacy_columns_only)
    except (OSError, ValueError) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
