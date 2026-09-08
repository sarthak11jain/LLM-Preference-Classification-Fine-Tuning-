"""Metrics and probability validation."""

import numpy as np
from sklearn.metrics import log_loss


def validate_probabilities(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected an N x 3 probability matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one")


def multiclass_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    validate_probabilities(probabilities)
    return float(log_loss(labels, probabilities, labels=[0, 1, 2]))


def average_fold_probabilities(fold_probabilities: list[np.ndarray], weights=None) -> np.ndarray:
    if not fold_probabilities:
        raise ValueError("At least one fold prediction is required")
    stack = np.stack(fold_probabilities, axis=0)
    if weights is None:
        result = np.mean(stack, axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(fold_probabilities),) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("weights must be non-negative and match the number of folds")
        result = np.average(stack, axis=0, weights=weights)
    result = np.clip(result, 0.0, None)
    result /= result.sum(axis=1, keepdims=True)
    validate_probabilities(result)
    return result
