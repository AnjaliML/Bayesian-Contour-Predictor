#!/usr/bin/env python3
"""Classify drop occurrence from a simple size law in generic x/y coordinates."""

from __future__ import annotations

import argparse
import math
from typing import Sequence


DEFAULT_SIZE_TOLERANCE = 5e-3
SIZE_SCALE = 0.2


def critical_y(x: float) -> float:
    return 0.0326 - 0.0398 * math.exp(-0.348 * x)


def drop_size(x: float, y: float) -> float:
    y_c = critical_y(x)
    if y_c <= 0 or y < 0:
        return 0.0
    raw_size = SIZE_SCALE * (1.0 - math.sqrt(y / y_c))
    return max(0.0, raw_size)


def size_threshold_y(x: float, tolerance: float = DEFAULT_SIZE_TOLERANCE) -> float:
    if tolerance < 0:
        raise ValueError("size tolerance cannot be negative")
    fraction = 1.0 - tolerance / SIZE_SCALE
    if fraction <= 0:
        return 0.0
    return critical_y(x) * fraction * fraction


def classify_drop(x: float, y: float, tolerance: float = DEFAULT_SIZE_TOLERANCE) -> int:
    return 1 if drop_size(x, y) > tolerance + 1e-12 else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify id from drop size, using x as radius ratio and y as Oh."
    )
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument(
        "--size-tolerance",
        type=float,
        default=DEFAULT_SIZE_TOLERANCE,
        help="Positive size threshold above which id is 1.",
    )
    parser.add_argument(
        "--print-size",
        action="store_true",
        help="Print the continuous size instead of the binary id.",
    )
    args = parser.parse_args(argv)
    if args.size_tolerance < 0:
        parser.error("--size-tolerance cannot be negative")

    if args.print_size:
        print(f"{drop_size(args.x, args.y):.12g}")
    else:
        print(classify_drop(args.x, args.y, args.size_tolerance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
