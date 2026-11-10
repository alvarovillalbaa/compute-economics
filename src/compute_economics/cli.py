"""Estimate explicit workloads without provider, planner, or Lab dependencies."""
import argparse
import json
import sys

from . import estimate_from_spec, format_estimate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="compute-economics")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("estimate-cost").add_argument("spec")
    args = parser.parse_args(argv)
    try:
        with open(args.spec, encoding="utf-8") as handle:
            spec = json.load(handle)
        print(format_estimate(estimate_from_spec(spec)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
