"""Competition submission formatting and validation."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import LABEL_COLUMNS
from .evaluation import validate_probabilities


def build_submission(ids: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    validate_probabilities(probabilities)
    if len(ids) != len(probabilities):
        raise ValueError("IDs and predictions must have the same length")
    result = pd.DataFrame({"id": ids.to_numpy()})
    result[list(LABEL_COLUMNS)] = probabilities
    return result


def write_submission(ids: pd.Series, probabilities: np.ndarray, path: str | Path) -> None:
    submission = build_submission(ids, probabilities)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
