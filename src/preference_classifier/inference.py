"""Inference and fold-ensemble helpers."""

import numpy as np

from .augmentation import remap_swapped_probabilities
from .evaluation import average_fold_probabilities, validate_probabilities


def combine_swap_tta(original: np.ndarray, swapped: np.ndarray) -> np.ndarray:
    validate_probabilities(original)
    validate_probabilities(swapped)
    if original.shape != swapped.shape:
        raise ValueError("original and swapped predictions must have equal shape")
    result = (original + remap_swapped_probabilities(swapped)) / 2.0
    validate_probabilities(result)
    return result


def ensemble_folds(fold_predictions: list[np.ndarray], weights=None) -> np.ndarray:
    return average_fold_probabilities(fold_predictions, weights=weights)
