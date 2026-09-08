import numpy as np
import pandas as pd
import pytest

from preference_classifier.data import validate_schema
from preference_classifier.evaluation import (
    average_fold_probabilities,
    multiclass_log_loss,
)
from preference_classifier.splits import grouped_folds
from preference_classifier.submission import build_submission
from preference_classifier.training import average_oof_predictions


def labeled_frame():
    return pd.DataFrame(
        {
            "prompt": ["p1", "p1", "p2", "p3", "p4", "p5"],
            "response_a": ["a"] * 6,
            "response_b": ["b"] * 6,
            "winner_model_a": [1, 0, 0, 0, 1, 0],
            "winner_model_b": [0, 1, 1, 0, 0, 1],
            "winner_tie": [0, 0, 0, 1, 0, 0],
        }
    )


def test_grouped_folds_keep_prompts_together():
    frame = labeled_frame()
    for train_idx, validation_idx in grouped_folds(frame, n_splits=3):
        assert set(frame.iloc[train_idx].prompt).isdisjoint(set(frame.iloc[validation_idx].prompt))


def test_submission_and_metrics_validate_probability_contract():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    submission = build_submission(pd.Series(["a", "b"]), probabilities)
    assert list(submission.columns) == ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert multiclass_log_loss(np.array([0, 2]), probabilities) > 0
    assert np.allclose(average_fold_probabilities([probabilities, probabilities]), probabilities)


def test_oof_requires_every_row():
    values = np.zeros((2, 3))
    with pytest.raises(ValueError):
        average_oof_predictions(values, [np.array([True, False])])


def test_schema_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(pd.DataFrame({"prompt": ["p"]}), training=True)
