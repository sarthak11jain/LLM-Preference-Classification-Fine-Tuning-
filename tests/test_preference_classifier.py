import numpy as np
import pandas as pd

from preference_classifier.augmentation import swap_rows, swap_tta
from preference_classifier.data import labels_from_frame, prompt_groups, validate_schema
from preference_classifier.evaluation import (
    average_fold_probabilities,
    multiclass_log_loss,
)


def frame():
    return pd.DataFrame(
        {
            "prompt": ["same", "same", "other"],
            "response_a": ["a1", "a2", "a3"],
            "response_b": ["b1", "b2", "b3"],
            "winner_model_a": [1.0, 0.0, 0.0],
            "winner_model_b": [0.0, 1.0, 0.0],
            "winner_tie": [0.0, 0.0, 1.0],
        }
    )


def test_schema_labels_and_prompt_groups():
    data = frame()
    validate_schema(data, training=True)
    assert labels_from_frame(data).tolist() == [0, 1, 2]
    assert prompt_groups(data).tolist() == [0, 0, 1]


def test_swap_changes_positions_and_a_b_labels():
    swapped = swap_rows(frame(), training=True)
    assert swapped.loc[0, "response_a"] == "b1"
    assert swapped.loc[0, "response_b"] == "a1"
    assert swapped.loc[0, "winner_model_a"] == 0.0
    assert swapped.loc[0, "winner_model_b"] == 1.0


def test_swap_tta_maps_a_and_b_back():
    result = swap_tta(np.array([[0.8, 0.1, 0.1]]), np.array([[0.1, 0.8, 0.1]]))
    np.testing.assert_allclose(result, [[0.8, 0.1, 0.1]])


def test_metrics_and_fold_averaging():
    probs = average_fold_probabilities([np.array([[0.8, 0.1, 0.1]]), np.array([[0.6, 0.2, 0.2]])])
    np.testing.assert_allclose(probs, [[0.7, 0.15, 0.15]])
    assert multiclass_log_loss(np.array([0]), probs) > 0
