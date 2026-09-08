"""Small public CLI for data validation and deterministic demonstrations."""

import argparse

import pandas as pd

from .augmentation import swap_rows, swap_tta
from .baselines import constant_probabilities
from .config import load_yaml
from .data import read_csv
from .submission import write_submission


def main() -> int:
    parser = argparse.ArgumentParser(prog="preference-classifier")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-data")
    validate.add_argument("--path", required=True)
    config = sub.add_parser("show-config")
    config.add_argument("--path", required=True)
    baseline = sub.add_parser("make-constant-submission")
    baseline.add_argument("--path", required=True, help="CSV containing prompt, response_a, response_b, and id")
    baseline.add_argument("--output", required=True)
    sub.add_parser("demo-swap")
    args = parser.parse_args()
    if args.command == "validate-data":
        frame = read_csv(args.path, training=True)
        print(f"Validated {len(frame):,} labeled rows")
    elif args.command == "show-config":
        print(load_yaml(args.path).to_dict())
    elif args.command == "make-constant-submission":
        frame = pd.read_csv(args.path)
        required = {"id", "prompt", "response_a", "response_b"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        write_submission(frame["id"], constant_probabilities(frame), args.output)
        print(f"Wrote {len(frame):,} predictions to {args.output}")
    else:
        frame = pd.DataFrame({"prompt": ["p"], "response_a": ["a"], "response_b": ["b"], "winner_model_a": [1.0], "winner_model_b": [0.0], "winner_tie": [0.0]})
        print(swap_rows(frame, training=True).to_dict(orient="records")[0])
        print(swap_tta([[0.8, 0.1, 0.1]], [[0.1, 0.8, 0.1]]).tolist()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
