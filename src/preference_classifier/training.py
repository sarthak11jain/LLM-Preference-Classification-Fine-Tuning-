"""Framework-agnostic training helpers used by the model workflows."""

import numpy as np


def cross_entropy_loss(label_smoothing: float = 0.0):
    """Create the loss used by the PyTorch fine-tuning workflows."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra for PyTorch training") from exc
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when the optional dependency is present."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def average_oof_predictions(oof: np.ndarray, fold_masks: list[np.ndarray]) -> np.ndarray:
    """Average predictions only over rows that have a validation prediction."""
    values = np.asarray(oof, dtype=float)
    mask = np.zeros(values.shape[0], dtype=bool)
    for fold_mask in fold_masks:
        mask |= np.asarray(fold_mask, dtype=bool)
    if not mask.all():
        raise ValueError("OOF predictions are incomplete")
    return values
