#!/usr/bin/env python3
"""Benchmark theory-blind acquisition; use known contours only for scoring."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import propose_next_sweep as sweep  # noqa: E402


Boundary = Callable[[float], float]


@dataclass(frozen=True)
class Profile:
    name: str
    contour_fit: str
    grid_size: int
    posterior_samples: int
    transition_width: float
    length_scale_x: float


PROFILES = {
    profile.name: profile
    for profile in (
        Profile("current", "adaptive-linear", 42, 16, 0.05, 0.18),
        Profile("balanced", "adaptive-linear", 27, 4, 0.04, 0.18),
        Profile("fast", "adaptive-linear", 21, 0, 0.04, 0.18),
        Profile("local-linear", "local-linear", 21, 0, 0.04, 0.18),
    )
}


def built_in_boundary(u: float) -> float:
    x = 10 ** (2.0 * u)
    y = 0.0326 - 0.0398 * math.exp(-0.348 * x)
    return (math.log10(y) + 3.0) / 2.0


BOUNDARIES: dict[str, Boundary] = {
    "built-in-holdout": built_in_boundary,
    "log-linear": lambda u: 0.22 + 0.52 * u,
    "sigmoid-knee": lambda u: 0.20 + 0.56 / (1.0 + math.exp(-10.0 * (u - 0.5))),
    "wavy": lambda u: 0.46 + 0.22 * math.sin(2.0 * math.pi * (u - 0.15)),
    "steep-edge": lambda u: 0.16 + 0.58 * math.sqrt(u),
}


DOMAIN = sweep.Domain(x_min=1.0, x_max=100.0, y_min=0.001, y_max=0.1)


def xy_from_unit(u: float, v: float) -> tuple[float, float]:
    return 10 ** (2.0 * u), 10 ** (-3.0 + 2.0 * v)


def unit_from_xy(x: float, y: float) -> tuple[float, float]:
    return math.log10(x) / 2.0, (math.log10(y) + 3.0) / 2.0


def model_config(profile: Profile) -> sweep.ModelConfig:
    return sweep.ModelConfig(
        mode="monotone-y",
        monotone_direction="decreasing",
        contour_fit=profile.contour_fit,
        x_scale="log10",
        y_scale="log10",
        transition_width=profile.transition_width,
        label_noise=0.005,
        length_scale_x=profile.length_scale_x,
        length_scale_y=0.4,
        prior_alpha=1.0,
        prior_beta=1.0,
        grid_size=profile.grid_size,
        posterior_samples=profile.posterior_samples,
    )


def observed_label(
    u: float,
    v: float,
    boundary: Boundary,
    *,
    noise_rate: float,
    rng: random.Random,
) -> int:
    label = int(v < boundary(u))
    return 1 - label if rng.random() < noise_rate else label


def initial_design(
    boundary: Boundary,
    *,
    count: int,
    noise_rate: float,
    seed: int,
) -> list[sweep.Observation]:
    rng = random.Random(seed)
    u_slots = list(range(count))
    v_slots = list(range(count))
    rng.shuffle(u_slots)
    rng.shuffle(v_slots)
    observations: list[sweep.Observation] = []
    for index, (u_slot, v_slot) in enumerate(zip(u_slots, v_slots), start=1):
        u = (u_slot + 0.5) / count
        v = (v_slot + 0.5) / count
        x, y = xy_from_unit(u, v)
        observations.append(
            sweep.Observation(
                str(index),
                x,
                y,
                observed_label(u, v, boundary, noise_rate=noise_rate, rng=rng),
            )
        )
    return observations


def truth_metrics(
    observations: Sequence[sweep.Observation],
    boundary: Boundary,
    config: sweep.ModelConfig,
    *,
    points: int,
    boundary_fraction: float = 0.12,
) -> dict[str, float]:
    aggregates = sweep.aggregate_observations(observations)
    squared: list[float] = []
    edge_squared: list[float] = []
    absolute: list[float] = []
    for index in range(points):
        u = index / (points - 1) if points > 1 else 0.5
        x, _ = xy_from_unit(u, 0.5)
        predicted = sweep.estimate_y_c(x, aggregates, DOMAIN, config)
        _, predicted_v = unit_from_xy(x, predicted)
        error = predicted_v - boundary(u)
        squared.append(error * error)
        absolute.append(abs(error))
        if u <= boundary_fraction or u >= 1.0 - boundary_fraction:
            edge_squared.append(error * error)
    return {
        "sse": sum(squared),
        "rmse": math.sqrt(sum(squared) / len(squared)),
        "edge_sse": sum(edge_squared),
        "edge_rmse": math.sqrt(sum(edge_squared) / len(edge_squared)),
        "max_error": max(absolute),
    }


def run_campaign(
    profile: Profile,
    boundary: Boundary,
    *,
    seed: int,
    noise_rate: float,
    initial_points: int,
    batch_size: int,
    iterations: int,
    eval_points: int,
) -> dict[str, float]:
    config = model_config(profile)
    observations = initial_design(
        boundary, count=initial_points, noise_rate=noise_rate, seed=seed
    )
    label_rng = random.Random(seed + 1_000_000)
    metrics_by_iteration: list[dict[str, float]] = []
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        proposals = sweep.propose_next_batch(
            observations,
            domain=DOMAIN,
            config=config,
            n_simulations=batch_size,
            n_new=batch_size if noise_rate == 0 else None,
            n_repeats=0 if noise_rate == 0 else None,
            seed=seed + iteration,
        )
        for proposal in proposals:
            u, v = unit_from_xy(proposal.x, proposal.y)
            observations.append(
                sweep.Observation(
                    str(len(observations) + 1),
                    proposal.x,
                    proposal.y,
                    observed_label(
                        u, v, boundary, noise_rate=noise_rate, rng=label_rng
                    ),
                )
            )
        metrics_by_iteration.append(
            truth_metrics(observations, boundary, config, points=eval_points)
        )
    elapsed = time.perf_counter() - started
    final = metrics_by_iteration[-1]
    return {
        **final,
        "sse_auc": sum(item["sse"] for item in metrics_by_iteration) / iterations,
        "seconds": elapsed,
        "simulations": float(len(observations)),
    }


def parse_csv_values(value: str, converter: Callable[[str], object]) -> list[object]:
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--families", default=",".join(BOUNDARIES))
    parser.add_argument("--seeds", default="11")
    parser.add_argument("--noise-rates", default="0")
    parser.add_argument("--initial-points", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-points", type=int, default=81)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_names = [str(value) for value in parse_csv_values(args.profiles, str)]
    family_names = [str(value) for value in parse_csv_values(args.families, str)]
    seeds = [int(value) for value in parse_csv_values(args.seeds, int)]
    noise_rates = [float(value) for value in parse_csv_values(args.noise_rates, float)]
    unknown_profiles = sorted(set(profile_names) - set(PROFILES))
    unknown_families = sorted(set(family_names) - set(BOUNDARIES))
    if unknown_profiles or unknown_families:
        raise SystemExit(
            f"unknown profiles={unknown_profiles or 'none'}, families={unknown_families or 'none'}"
        )

    rows: list[dict[str, object]] = []
    for profile_name in profile_names:
        for family_name in family_names:
            for noise_rate in noise_rates:
                for seed in seeds:
                    result = run_campaign(
                        PROFILES[profile_name],
                        BOUNDARIES[family_name],
                        seed=seed,
                        noise_rate=noise_rate,
                        initial_points=args.initial_points,
                        batch_size=args.batch_size,
                        iterations=args.iterations,
                        eval_points=args.eval_points,
                    )
                    row = {
                        "profile": profile_name,
                        "family": family_name,
                        "seed": seed,
                        "noise_rate": noise_rate,
                        **result,
                    }
                    rows.append(row)
                    print(
                        f"{profile_name:12s} {family_name:16s} seed={seed} "
                        f"SSE={result['sse']:.6g} edge={result['edge_sse']:.6g} "
                        f"time={result['seconds']:.3f}s"
                    )

    print("\nProfile summary (median across requested families/seeds/noise rates):")
    for profile_name in profile_names:
        selected = [row for row in rows if row["profile"] == profile_name]
        print(
            f"{profile_name:12s} "
            f"SSE={statistics.median(float(row['sse']) for row in selected):.6g} "
            f"edge={statistics.median(float(row['edge_sse']) for row in selected):.6g} "
            f"AUC={statistics.median(float(row['sse_auc']) for row in selected):.6g} "
            f"time={sum(float(row['seconds']) for row in selected):.3f}s"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
