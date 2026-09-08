"""Data schema and label utilities."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import INPUT_COLUMNS, LABEL_COLUMNS


def validate_schema(frame: pd.DataFrame, *, training: bool = False) -> None:
    required = set(INPUT_COLUMNS) | (set(LABEL_COLUMNS) if training else set())
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[list(INPUT_COLUMNS)].isna().any().any():
        raise ValueError("Input text columns must not contain null values")
    if training:
        values = frame[list(LABEL_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Label probabilities must be finite and non-negative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Each target row must sum to one")


def read_csv(path: str | Path, *, training: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_schema(frame, training=training)
    return frame


def labels_from_frame(frame: pd.DataFrame) -> np.ndarray:
    validate_schema(frame, training=True)
    return frame[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32).argmax(axis=1)


def prompt_groups(frame: pd.DataFrame) -> np.ndarray:
    """Return stable group IDs so identical prompts cannot cross a fold boundary."""
    return pd.factorize(frame["prompt"].astype(str), sort=False)[0].astype(np.int64)
