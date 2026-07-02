#!/usr/bin/env python3
import argparse
import math


def oh_c(rr):
    return 0.0326 - 0.0398 * math.exp(-0.348 * rr)


parser = argparse.ArgumentParser()
parser.add_argument("Oh", type=float)
parser.add_argument("Rr", type=float)
args = parser.parse_args()

print(1 if args.Oh < oh_c(args.Rr) else 0)
