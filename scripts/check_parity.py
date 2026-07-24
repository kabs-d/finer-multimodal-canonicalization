#!/usr/bin/env python3
"""Compare standalone centered metrics with the corresponding author-code output."""

import argparse
import json
from pathlib import Path


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from flatten(child, child_prefix)
    else:
        yield prefix, float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--standalone", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    upstream = json.loads(args.upstream.read_text(encoding="utf-8"))["mean"]
    standalone = json.loads(args.standalone.read_text(encoding="utf-8"))["mean"]
    upstream_flat = dict(flatten(upstream))
    standalone_flat = dict(flatten(standalone))
    if upstream_flat.keys() != standalone_flat.keys():
        raise SystemExit("metric keys differ")
    differences = {
        key: abs(upstream_flat[key] - standalone_flat[key])
        for key in upstream_flat
    }
    largest_key = max(differences, key=differences.get)
    largest = differences[largest_key]
    print(
        json.dumps(
            {
                "passed": largest <= args.tolerance,
                "tolerance": args.tolerance,
                "maximum_absolute_difference": largest,
                "maximum_difference_metric": largest_key,
            },
            indent=2,
        )
    )
    if largest > args.tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

