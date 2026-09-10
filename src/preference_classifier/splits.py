"""Validation split strategies used by the training pipeline."""

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import labels_from_frame, prompt_groups


def grouped_folds(frame, n_splits: int = 3):
    labels = labels_from_frame(frame)
    groups = prompt_groups(frame)
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(frame, labels, groups=groups)


def id_modulo_split(frame, modulus: int = 5, remainder: int = 0):
    """Return train and validation indices for the deterministic ID split."""
    if modulus < 2 or not 0 <= remainder < modulus:
        raise ValueError("modulus must be >= 2 and remainder must be in [0, modulus)")
    if "id" not in frame.columns:
        raise ValueError("id_modulo split requires an id column")
    validation = frame["id"].astype(int).mod(modulus).eq(remainder).to_numpy()
    if not validation.any() or validation.all():
        raise ValueError("id_modulo split must produce both training and validation rows")
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def configured_splits(frame, config):
    """Dispatch to the split strategy selected by an ExperimentConfig."""
    if config.split_strategy == "id_modulo":
        yield id_modulo_split(frame, config.split_modulus, config.split_remainder)
    elif config.split_strategy == "grouped":
        yield from grouped_folds(frame, config.n_folds)
    else:
        raise ValueError(f"Unknown split strategy: {config.split_strategy}")
