import numpy as np

from preference_classifier.calibration import (
    brier_score,
    expected_calibration_error,
    probability_histograms,
    reliability_bins,
    temperature_scale,
)


def test_calibration_metrics_and_temperature_scaling():
    labels = np.array([0, 1, 2])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    assert brier_score(labels, probabilities) > 0
    assert expected_calibration_error(labels, probabilities) >= 0
    assert sum(row["count"] for row in reliability_bins(labels, probabilities)) == 3
    scaled = temperature_scale(probabilities, 2.0)
    np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
    assert sum(probability_histograms(probabilities)["class_0"]) == 3
