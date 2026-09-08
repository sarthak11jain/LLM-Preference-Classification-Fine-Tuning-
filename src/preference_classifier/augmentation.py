"""Response-order augmentation and probability remapping."""

import numpy as np
import pandas as pd

from . import LABEL_COLUMNS

SWAP_LABEL_MAP = np.array([1, 0, 2], dtype=np.int64)


def swap_rows(frame: pd.DataFrame, *, training: bool = False) -> pd.DataFrame:
    """Swap response positions and, for training data, swap A/B target labels."""
    result = frame.copy()
    result["response_a"], result["response_b"] = frame["response_b"].to_numpy(), frame["response_a"].to_numpy()
    if training:
        result["winner_model_a"], result["winner_model_b"] = (
            frame["winner_model_b"].to_numpy(),
            frame["winner_model_a"].to_numpy(),
        )
        result["winner_tie"] = frame["winner_tie"].to_numpy()
    return result


def augment_with_swaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Double a labeled training frame while preserving the original row order first."""
    return pd.concat([frame, swap_rows(frame, training=True)], ignore_index=True)


def remap_swapped_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(LABEL_COLUMNS):
        raise ValueError("Expected an N x 3 probability matrix")
    return probs[:, SWAP_LABEL_MAP]


def swap_tta(original: np.ndarray, swapped: np.ndarray) -> np.ndarray:
    result = 0.5 * (np.asarray(original) + remap_swapped_probabilities(swapped))
    result = np.clip(result, 0.0, None)
    return result / result.sum(axis=1, keepdims=True)
