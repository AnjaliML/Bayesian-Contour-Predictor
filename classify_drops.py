#!/usr/bin/env python3
import argparse
import math
from typing import Sequence


def oh_c(rr):
    return 0.0326 - 0.0398 * math.exp(-0.348 * rr)


def classify_drop(oh, rr):
    return 1 if oh < oh_c(rr) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("Oh", type=float)
    parser.add_argument("Rr", type=float)
    args = parser.parse_args(argv)
    print(classify_drop(args.Oh, args.Rr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
