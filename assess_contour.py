#!/usr/bin/env python3
"""Export a process-agnostic contour and assess theory-blind convergence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import propose_next_sweep as sweep


CONTOUR_COLUMNS = ["x", "y_pred", "y_q05", "y_q95", "y_std"]


def normalized_value(value: float, lower: float, upper: float, scale: str) -> float:
    lower_t = sweep.transformed_value(lower, scale, "value")
    upper_t = sweep.transformed_value(upper, scale, "value")
    value_t = sweep.transformed_value(value, scale, "value")
    return (value_t - lower_t) / max(upper_t - lower_t, 1e-12)


def build_contour(
    aggregates: Sequence[sweep.AggregatePoint],
    domain: sweep.Domain,
    config: sweep.ModelConfig,
    *,
    points: int,
    seed: int,
) -> list[dict[str, float]]:
    lower = sweep.transformed_value(domain.x_min, config.x_scale, "x")
    upper = sweep.transformed_value(domain.x_max, config.x_scale, "x")
    rng = random.Random(seed)
    contour: list[dict[str, float]] = []
    for x_t in sweep.linspace(lower, upper, points):
        x = sweep.inverse_transformed_value(x_t, config.x_scale)
        estimate = sweep.contour_estimate(x, aggregates, domain, config, rng)
        contour.append(
            {
                "x": x,
                "y_pred": estimate.y_c_pred,
                "y_q05": estimate.y_c_q05,
                "y_q95": estimate.y_c_q95,
                "y_std": estimate.y_c_std,
            }
        )
    return contour


def contour_movement(
    previous: Sequence[dict[str, float]],
    current: Sequence[dict[str, float]],
    domain: sweep.Domain,
    config: sweep.ModelConfig,
    *,
    boundary_fraction: float,
) -> dict[str, float] | None:
    if not previous or len(previous) != len(current):
        return None
    deltas: list[float] = []
    edge_deltas: list[float] = []
    for index, (before, after) in enumerate(zip(previous, current)):
        before_y = normalized_value(
            float(before["y_pred"]), domain.y_min, domain.y_max, config.y_scale
        )
        after_y = normalized_value(
            float(after["y_pred"]), domain.y_min, domain.y_max, config.y_scale
        )
        delta = after_y - before_y
        deltas.append(delta)
        fraction = index / (len(current) - 1) if len(current) > 1 else 0.5
        if fraction <= boundary_fraction or fraction >= 1.0 - boundary_fraction:
            edge_deltas.append(delta)
    sse = sum(delta * delta for delta in deltas)
    edge_sse = sum(delta * delta for delta in edge_deltas)
    return {
        "sse": sse,
        "mse": sse / len(deltas),
        "rmse": math.sqrt(sse / len(deltas)),
        "max": max(abs(delta) for delta in deltas),
        "edge_sse": edge_sse,
        "edge_rmse": math.sqrt(edge_sse / len(edge_deltas)) if edge_deltas else 0.0,
        "edge_max": max((abs(delta) for delta in edge_deltas), default=0.0),
        "points": float(len(deltas)),
    }


def contour_resolution(
    aggregates: Sequence[sweep.AggregatePoint],
    domain: sweep.Domain,
    config: sweep.ModelConfig,
) -> dict[str, Any]:
    if config.mode != "monotone-y":
        return {
            "supported": False,
            "reason": "bracket convergence requires --mode monotone-y",
            "bracketed_anchors": 0,
            "max_x_gap": 1.0,
            "max_y_bracket_width": 1.0,
            "mean_y_bracket_width": 1.0,
            "edge_bracketed": {"lower": False, "upper": False},
        }
    anchors = sweep.bracket_midpoint_candidates(aggregates, domain, config)
    lower_x = sweep.transformed_value(domain.x_min, config.x_scale, "x")
    upper_x = sweep.transformed_value(domain.x_max, config.x_scale, "x")
    x_span = max(upper_x - lower_x, 1e-12)
    anchor_x = sorted(
        sweep.transformed_value(x, config.x_scale, "x") for x, _, _ in anchors
    )
    gaps = [
        right - left
        for left, right in zip([lower_x, *anchor_x], [*anchor_x, upper_x])
    ]
    edge_tolerance = x_span * 1e-9
    edge_bracketed = {
        "lower": any(abs(value - lower_x) <= edge_tolerance for value in anchor_x),
        "upper": any(abs(value - upper_x) <= edge_tolerance for value in anchor_x),
    }
    return {
        "supported": True,
        "bracketed_anchors": len(anchors),
        "max_x_gap": max(gaps, default=1.0) / x_span,
        "max_y_bracket_width": max((width for _, _, width in anchors), default=1.0),
        "mean_y_bracket_width": (
            sum(width for _, _, width in anchors) / len(anchors) if anchors else 1.0
        ),
        "edge_bracketed": edge_bracketed,
    }


def write_contour(path: Path, contour: Sequence[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTOUR_COLUMNS)
        writer.writeheader()
        writer.writerows(contour)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a learned contour and assess convergence without theoretical truth."
    )
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--case-col", default="caseId")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--label-col", default="id")
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--mode", choices=["generic", "monotone-y"], default="monotone-y")
    parser.add_argument(
        "--monotone-direction", choices=["decreasing", "increasing"], default="decreasing"
    )
    parser.add_argument(
        "--contour-fit",
        choices=["local-constant", "local-linear", "adaptive-linear"],
        default="local-linear",
    )
    parser.add_argument("--x-scale", choices=["linear", "log10"], default="linear")
    parser.add_argument("--y-scale", choices=["linear", "log10"], default="linear")
    parser.add_argument("--transition-width", type=float, default=0.04)
    parser.add_argument("--label-noise", type=float, default=0.005)
    parser.add_argument("--length-scale-x", type=float)
    parser.add_argument("--length-scale-y", type=float)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--posterior-samples", type=int, default=0)
    parser.add_argument("--contour-points", type=int, default=101)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boundary-fraction", type=float, default=0.12)
    parser.add_argument("--movement-rmse-tolerance", type=float, default=0.0025)
    parser.add_argument("--movement-max-tolerance", type=float, default=0.01)
    parser.add_argument("--movement-edge-tolerance", type=float, default=0.005)
    parser.add_argument("--max-x-gap", type=float, default=0.15)
    parser.add_argument("--max-y-bracket-width", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--allow-unbracketed-edges", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.x_min >= args.x_max or args.y_min >= args.y_max:
        raise ValueError("axis minima must be less than maxima")
    if args.contour_points < 2:
        raise ValueError("--contour-points must be at least 2")
    if not 0 <= args.boundary_fraction <= 0.5:
        raise ValueError("--boundary-fraction must be between 0 and 0.5")
    if args.patience < 1 or args.min_iterations < 0:
        raise ValueError("patience must be positive and min iterations non-negative")
    if args.length_scale_x is not None and args.length_scale_x <= 0:
        raise ValueError("--length-scale-x must be positive")
    if args.length_scale_y is not None and args.length_scale_y <= 0:
        raise ValueError("--length-scale-y must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        observations, _ = sweep.read_csv_files(
            args.csv_files,
            case_col=args.case_col,
            x_col=args.x_col,
            y_col=args.y_col,
            label_col=args.label_col,
        )
        observation_signature = hashlib.sha256(
            json.dumps(
                sorted(
                    (item.case_id, item.x, item.y, item.label)
                    for item in observations
                ),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        domain = sweep.Domain(args.x_min, args.x_max, args.y_min, args.y_max)
        sweep.transformed_span(args.x_min, args.x_max, args.x_scale, "x")
        sweep.transformed_span(args.y_min, args.y_max, args.y_scale, "y")
        config = sweep.ModelConfig(
            mode=args.mode,
            monotone_direction=args.monotone_direction,
            contour_fit=args.contour_fit,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            transition_width=args.transition_width,
            label_noise=args.label_noise,
            length_scale_x=args.length_scale_x
            or sweep.default_length_scale(
                sweep.transformed_span(args.x_min, args.x_max, args.x_scale, "x")
            ),
            length_scale_y=args.length_scale_y
            or sweep.default_length_scale(
                sweep.transformed_span(args.y_min, args.y_max, args.y_scale, "y")
            ),
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
            grid_size=args.grid_size,
            posterior_samples=args.posterior_samples,
        )
        config_payload = {
            "mode": config.mode,
            "monotone_direction": config.monotone_direction,
            "contour_fit": config.contour_fit,
            "x_scale": config.x_scale,
            "y_scale": config.y_scale,
            "transition_width": config.transition_width,
            "label_noise": config.label_noise,
            "length_scale_x": config.length_scale_x,
            "length_scale_y": config.length_scale_y,
            "grid_size": config.grid_size,
            "posterior_samples": config.posterior_samples,
            "contour_points": args.contour_points,
        }
        previous: dict[str, Any] = {}
        if args.state.exists():
            previous = json.loads(args.state.read_text(encoding="utf-8"))
        iteration = args.iteration
        if iteration is None:
            iteration = int(previous.get("iteration", -1)) + 1
        if (
            previous.get("iteration") == iteration
            and previous.get("config") == config_payload
            and previous.get("observation_signature") == observation_signature
            and previous.get("contour")
        ):
            previous["contour_file"] = str(args.outfile)
            write_contour(args.outfile, previous["contour"])
            write_json(args.state, previous)
            print(
                json.dumps(
                    {key: value for key, value in previous.items() if key != "contour"}
                )
            )
            return 0
        contour = build_contour(
            sweep.aggregate_observations(observations),
            domain,
            config,
            points=args.contour_points,
            seed=args.seed + iteration,
        )
        movement = None
        if not previous.get("config") or previous.get("config") == config_payload:
            movement = contour_movement(
                previous.get("contour", []),
                contour,
                domain,
                config,
                boundary_fraction=args.boundary_fraction,
            )
        resolution = contour_resolution(
            sweep.aggregate_observations(observations), domain, config
        )
        movement_stable = bool(
            movement
            and movement["rmse"] <= args.movement_rmse_tolerance
            and movement["max"] <= args.movement_max_tolerance
            and movement["edge_max"] <= args.movement_edge_tolerance
        )
        edges_resolved = args.allow_unbracketed_edges or all(
            resolution["edge_bracketed"].values()
        )
        resolution_ready = bool(
            resolution["supported"]
            and resolution["max_x_gap"] <= args.max_x_gap
            and resolution["max_y_bracket_width"] <= args.max_y_bracket_width
            and edges_resolved
        )
        stable_checks = int(previous.get("stable_checks", 0)) + 1 if movement_stable else 0
        stalled_checks = (
            int(previous.get("stalled_checks", 0)) + 1
            if movement_stable and not resolution_ready
            else 0
        )
        if (
            iteration >= args.min_iterations
            and stable_checks >= args.patience
            and resolution_ready
        ):
            status = "converged"
        elif iteration >= args.min_iterations and stalled_checks >= args.patience:
            status = "stalled"
        else:
            status = "refining"
        state = {
            "status": status,
            "iteration": iteration,
            "stable_checks": stable_checks,
            "stalled_checks": stalled_checks,
            "movement_stable": movement_stable,
            "resolution_ready": resolution_ready,
            "movement": movement,
            "resolution": resolution,
            "observation_count": len(observations),
            "observation_signature": observation_signature,
            "contour_file": str(args.outfile),
            "config": config_payload,
            "contour": contour,
        }
        write_contour(args.outfile, contour)
        write_json(args.state, state)
        print(json.dumps({key: value for key, value in state.items() if key != "contour"}))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
