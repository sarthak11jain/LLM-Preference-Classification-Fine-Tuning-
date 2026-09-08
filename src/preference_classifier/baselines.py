"""Small, reproducible baselines used to sanity-check the task."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import labels_from_frame
from .preprocessing import format_comparison


def comparison_texts(frame) -> list[str]:
    return [
        format_comparison(prompt, response_a, response_b)
        for prompt, response_a, response_b in zip(
            frame["prompt"], frame["response_a"], frame["response_b"]
        )
    ]


def constant_probabilities(frame) -> np.ndarray:
    class_ids = labels_from_frame(frame)
    mean = np.bincount(class_ids, minlength=3).astype(float) / len(class_ids)
    return np.repeat(mean[None, :], len(frame), axis=0)


def fit_tfidf_logistic_regression(train_frame, validation_frame, max_features: int = 100_000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    train_x = vectorizer.fit_transform(comparison_texts(train_frame))
    validation_x = vectorizer.transform(comparison_texts(validation_frame))
    model = LogisticRegression(max_iter=300, multi_class="multinomial")
    model.fit(vectorizer.transform(comparison_texts(train_frame)), labels_from_frame(train_frame).argmax(axis=1))
    return vectorizer, model, model.predict_proba(validation_x)
