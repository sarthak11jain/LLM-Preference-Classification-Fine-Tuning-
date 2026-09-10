"""Small, reproducible baselines used to sanity-check the task."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import labels_from_frame
from .preprocessing import format_comparison


def comparison_texts(frame) -> list[str]:
    return [
        format_comparison(prompt, response_a, response_b)
        for prompt, response_a, response_b in zip(frame["prompt"], frame["response_a"], frame["response_b"])
    ]


def constant_probabilities(frame, prior: tuple[float, float, float] | None = None) -> np.ndarray:
    """Return a deterministic prior for every row.

    Test data has no winner columns, so the public baseline defaults to a
    uniform prior. A labeled frame may still be used to estimate a prior for
    local experiments when explicitly requested.
    """
    if prior is None:
        mean = np.full(3, 1 / 3, dtype=float)
    else:
        mean = np.asarray(prior, dtype=float)
        if mean.shape != (3,) or (mean < 0).any() or not np.isclose(mean.sum(), 1.0):
            raise ValueError("prior must contain three non-negative values summing to one")
    return np.repeat(mean[None, :], len(frame), axis=0)


def training_prior(frame) -> np.ndarray:
    """Estimate a class prior from a labeled training frame."""
    class_ids = labels_from_frame(frame)
    mean = np.bincount(class_ids, minlength=3).astype(float) / len(class_ids)
    return constant_probabilities(frame, tuple(mean))


def fit_tfidf_logistic_regression(train_frame, validation_frame, max_features: int = 100_000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    vectorizer.fit_transform(comparison_texts(train_frame))
    validation_x = vectorizer.transform(comparison_texts(validation_frame))
    model = LogisticRegression(max_iter=300, multi_class="multinomial")
    model.fit(vectorizer.transform(comparison_texts(train_frame)), labels_from_frame(train_frame))
    raw_probabilities = model.predict_proba(validation_x)
    probabilities = np.zeros((len(validation_frame), 3), dtype=float)
    probabilities[:, model.classes_.astype(int)] = raw_probabilities
    return vectorizer, model, probabilities
