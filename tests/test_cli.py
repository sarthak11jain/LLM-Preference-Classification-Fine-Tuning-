import json

import pandas as pd

from preference_classifier.cli import main


def test_cli_config_and_constant_submission(tmp_path, capsys):
    root = tmp_path / "project"
    root.mkdir()
    config = root / "config.yaml"
    config.write_text("model_name: demo/model\nmax_length: 128\n", encoding="utf-8")
    test_csv = root / "test.csv"
    pd.DataFrame({"id": [1], "prompt": ["p"], "response_a": ["a"], "response_b": ["b"]}).to_csv(test_csv, index=False)
    output = root / "nested" / "submission.csv"

    assert main(["show-config", "--path", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["max_length_train"] == 128
    assert main(["make-constant-submission", "--path", str(test_csv), "--output", str(output)]) == 0
    submission = pd.read_csv(output)
    assert submission.loc[0, "winner_tie"] == 1 / 3


def test_cli_cpu_training_and_demo(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    config = root / "config.yaml"
    config.write_text(
        "model_name: demo/model\nsplit_strategy: id_modulo\nsplit_modulus: 5\nsplit_remainder: 0\n",
        encoding="utf-8",
    )
    train = pd.DataFrame(
        {
            "id": range(10),
            "prompt": [f"p{i}" for i in range(10)],
            "response_a": ["good answer"] * 10,
            "response_b": ["bad answer"] * 10,
            "winner_model_a": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "winner_model_b": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "winner_tie": [0] * 10,
        }
    )
    train_path = root / "train.csv"
    train.to_csv(train_path, index=False)
    arguments = [
        "train",
        "--config",
        str(config),
        "--train",
        str(train_path),
        "--output-dir",
        str(root / "run"),
        "--backend",
        "tfidf",
    ]
    assert main(arguments) == 0
    assert (root / "run" / "metrics.json").exists()


def test_cli_validation_evaluation_calibration_prediction_and_swap(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    config = root / "config.yaml"
    config.write_text("model_name: demo/model\nmax_length: 128\n", encoding="utf-8")
    labels = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "prompt": ["p1", "p2", "p3"],
            "response_a": ["a"] * 3,
            "response_b": ["b"] * 3,
            "winner_model_a": [1, 0, 0],
            "winner_model_b": [0, 1, 0],
            "winner_tie": [0, 0, 1],
        }
    )
    labels_path = root / "labels.csv"
    labels.to_csv(labels_path, index=False)
    predictions = root / "predictions.csv"
    pd.DataFrame(
        {
            "winner_model_a": [0.8, 0.1, 0.1],
            "winner_model_b": [0.1, 0.8, 0.1],
            "winner_tie": [0.1, 0.1, 0.8],
        }
    ).to_csv(predictions, index=False)
    test_path = root / "test.csv"
    labels.drop(columns=["winner_model_a", "winner_model_b", "winner_tie"]).to_csv(test_path, index=False)

    assert main(["validate-data", "--path", str(labels_path)]) == 0
    assert main(["evaluate", "--labels", str(labels_path), "--predictions", str(predictions)]) == 0
    assert (
        main(
            [
                "calibrate",
                "--labels",
                str(labels_path),
                "--predictions",
                str(predictions),
                "--output-dir",
                str(root / "calibration"),
            ]
        )
        == 0
    )
    assert main(["predict", "--config", str(config), "--test", str(test_path), "--output", str(root / "pred.csv")]) == 0
    assert main(["demo-swap"]) == 0
    assert main(["demo", "--output-dir", str(root / "demo")]) == 0
