"""Leakage-aware grouped cross-validation."""

from sklearn.model_selection import GroupKFold

from .data import labels_from_frame, prompt_groups


def grouped_folds(frame, n_splits: int = 3):
    labels = labels_from_frame(frame)
    groups = prompt_groups(frame)
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(frame, labels, groups=groups)
