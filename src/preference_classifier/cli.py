"""Command-line interface for validation, demos, evaluation, and inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABEL_COLUMNS
from .augmentation import swap_rows, swap_tta
from .baselines import constant_probabilities
from .calibration import (
    brier_score,
    expected_calibration_error,
    probability_histograms,
    reliability_bins,
    temperature_scale,
    write_reliability_svg,
)
from .config import load_yaml
from .data import read_csv
from .evaluation import multiclass_log_loss, validate_probabilities
from .inference import combine_swap_tta
from .pipeline import make_demo_submission, run_tfidf_validation, run_transformer_prediction, run_transformer_training
from .submission import write_submission


def _probabilities(path: str | Path) -> np.ndarray:
    frame = pd.read_csv(path)
    missing = sorted(set(LABEL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    probabilities = frame[list(LABEL_COLUMNS)].to_numpy(dtype=float)
    validate_probabilities(probabilities)
    return probabilities


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True, help="Labeled training CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("tfidf", "transformer"), default="transformer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="Validate a competition CSV")
    validate.add_argument("--path", required=True)
    config = sub.add_parser("show-config", help="Load and display a YAML configuration")
    config.add_argument("--path", required=True)

    baseline = sub.add_parser("make-constant-submission", help="Write a uniform-prior submission")
    baseline.add_argument("--path", required=True, help="Unlabeled test CSV")
    baseline.add_argument("--output", required=True)

    train = sub.add_parser("train", help="Run a CPU baseline or optional transformer training")
    _add_training_args(train)

    predict = sub.add_parser("predict", help="Generate predictions from a saved run")
    predict.add_argument("--config", required=True)
    predict.add_argument("--test", required=True)
    predict.add_argument("--checkpoint", required=False)
    predict.add_argument("--output", required=True)
    predict.add_argument("--backend", choices=("constant", "transformer"), default="constant")

    evaluate = sub.add_parser("evaluate", help="Score a prediction CSV against labeled data")
    evaluate.add_argument("--labels", required=True, help="Labeled validation CSV")
    evaluate.add_argument("--predictions", required=True, help="CSV containing the three probabilities")

    calibrate = sub.add_parser("calibrate", help="Write calibration metrics and a reliability diagram")
    calibrate.add_argument("--labels", required=True)
    calibrate.add_argument("--predictions", required=True)
    calibrate.add_argument("--output-dir", required=True)
    calibrate.add_argument("--temperature", type=float, default=1.0)
    calibrate.add_argument("--swapped-predictions", help="Optional swapped-response probabilities for TTA comparison")

    demo = sub.add_parser("demo", help="Run the CPU-safe synthetic demonstration")
    demo.add_argument("--output-dir", default="outputs/demo")

    sub.add_parser("demo-swap", help="Demonstrate response swapping and TTA")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        frame = read_csv(args.path, training=True)
        print(f"Validated {len(frame):,} labeled rows")
    elif args.command == "show-config":
        print(json.dumps(load_yaml(args.path).to_dict(), indent=2, default=list))
    elif args.command == "make-constant-submission":
        frame = pd.read_csv(args.path)
        required = {"id", "prompt", "response_a", "response_b"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        write_submission(frame["id"], constant_probabilities(frame), args.output)
        print(f"Wrote {len(frame):,} predictions to {args.output}")
    elif args.command == "train":
        config = load_yaml(args.config)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.backend == "tfidf":
            result = run_tfidf_validation(args.train, config)
            (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
        else:
            result = run_transformer_training(args.train, config, output_dir)
            print(json.dumps(result, indent=2, default=list))
    elif args.command == "predict":
        load_yaml(args.config)
        test = pd.read_csv(args.test)
        required = {"id", "prompt", "response_a", "response_b"}
        missing = sorted(required - set(test.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if args.backend == "constant":
            write_submission(test["id"], constant_probabilities(test), args.output)
            print(f"Wrote {len(test):,} predictions to {args.output}")
        else:
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for transformer prediction")
            run_transformer_prediction(args.test, load_yaml(args.config), args.checkpoint, args.output)
            print(f"Wrote {len(test):,} predictions to {args.output}")
    elif args.command == "evaluate":
        labels = read_csv(args.labels, training=True)
        label_ids = labels[list(LABEL_COLUMNS)].to_numpy().argmax(axis=1)
        probabilities = _probabilities(args.predictions)
        score = multiclass_log_loss(
            label_ids,
            probabilities,
        )
        print(json.dumps({"metric": "log_loss", "score": score}, indent=2))
    elif args.command == "calibrate":
        labels = read_csv(args.labels, training=True)
        probabilities = _probabilities(args.predictions)
        label_ids = labels[list(LABEL_COLUMNS)].to_numpy().argmax(axis=1)
        rows = reliability_bins(label_ids, probabilities)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "log_loss": multiclass_log_loss(label_ids, probabilities),
            "brier_score": brier_score(label_ids, probabilities),
            "expected_calibration_error": expected_calibration_error(label_ids, probabilities),
            "temperature": args.temperature,
            "temperature_scaled_log_loss": multiclass_log_loss(
                label_ids, temperature_scale(probabilities, args.temperature)
            ),
            "bins": rows,
        }
        (output_dir / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_reliability_svg(rows, output_dir / "reliability.svg")
        (output_dir / "histograms.json").write_text(
            json.dumps(probability_histograms(probabilities), indent=2), encoding="utf-8"
        )
        from .calibration import write_probability_histogram_svg

        write_probability_histogram_svg(
            probability_histograms(probabilities), output_dir / "probability_histograms.svg"
        )
        if args.swapped_predictions:
            swapped = _probabilities(args.swapped_predictions)
            tta = combine_swap_tta(probabilities, swapped)
            report["swap_tta_log_loss"] = multiclass_log_loss(label_ids, tta)
            (output_dir / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "demo":
        root = Path(__file__).resolve().parents[2]
        demo_root = root / "data" / "demo"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        make_demo_submission(demo_root / "test.csv", output_dir / "submission.csv")
        validation = pd.read_csv(demo_root / "validation_predictions.csv")
        labels = read_csv(demo_root / "validation.csv", training=True)
        label_ids = labels[list(LABEL_COLUMNS)].to_numpy().argmax(axis=1)
        probabilities = validation[list(LABEL_COLUMNS)].to_numpy(dtype=float)
        uniform = np.full(probabilities.shape, 1 / 3, dtype=float)
        result = {
            "model": "Uniform constant prior",
            "metric": "log_loss",
            "score": multiclass_log_loss(label_ids, uniform),
            "scope": "synthetic CPU demo",
        }
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        report = {
            "log_loss": multiclass_log_loss(label_ids, probabilities),
            "brier_score": brier_score(label_ids, probabilities),
            "expected_calibration_error": expected_calibration_error(label_ids, probabilities),
        }
        (output_dir / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_reliability_svg(reliability_bins(label_ids, probabilities), output_dir / "reliability.svg")
        from .calibration import write_probability_histogram_svg

        write_probability_histogram_svg(
            probability_histograms(probabilities), output_dir / "probability_histograms.svg"
        )
        print(f"CPU demo complete: {output_dir}")
    else:
        frame = pd.DataFrame(
            {
                "prompt": ["p"],
                "response_a": ["a"],
                "response_b": ["b"],
                "winner_model_a": [1.0],
                "winner_model_b": [0.0],
                "winner_tie": [0.0],
            }
        )
        print(swap_rows(frame, training=True).to_dict(orient="records")[0])
        print(swap_tta([[0.8, 0.1, 0.1]], [[0.1, 0.8, 0.1]]).tolist()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
