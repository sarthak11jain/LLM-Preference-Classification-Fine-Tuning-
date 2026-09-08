import numpy as np
import pandas as pd

from preference_classifier.augmentation import SWAP_LABEL_MAP
from preference_classifier.baselines import constant_probabilities
from preference_classifier.config import ExperimentConfig
from preference_classifier.inference import combine_swap_tta
from preference_classifier.preprocessing import allocate_budgets, truncate_head_tail


def test_config_round_trip():
    config = ExperimentConfig(model_name="example/model", max_length=2048)
    assert ExperimentConfig.from_mapping(config.to_dict()) == config


def test_head_tail_and_budget_helpers():
    assert truncate_head_tail(list(range(10)), 4) == [0, 1, 8, 9]
    assert allocate_budgets([10, 10, 10], 9, [1, 1, 1]) == [3, 3, 3]


def test_constant_baseline_and_swap_tta():
    frame = pd.DataFrame({"prompt": ["p", "q"], "response_a": ["a", "b"], "response_b": ["b", "a"], "winner_model_a": [1, 0], "winner_model_b": [0, 1], "winner_tie": [0, 0]})
    probs = constant_probabilities(frame)
    assert np.allclose(probs, [[0.5, 0.5, 0], [0.5, 0.5, 0]])
    original = np.array([[0.7, 0.2, 0.1]])
    swapped = np.array([[0.2, 0.7, 0.1]])
    assert np.allclose(combine_swap_tta(original, swapped), original)
    assert np.array_equal(SWAP_LABEL_MAP, [1, 0, 2])
