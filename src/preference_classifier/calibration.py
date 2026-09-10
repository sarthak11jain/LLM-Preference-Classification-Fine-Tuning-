"""Probability calibration metrics and lightweight report generation."""

from pathlib import Path

import numpy as np

from .evaluation import validate_probabilities


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the multiclass Brier score."""
    validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=int)
    one_hot = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Build top-class reliability bins for a compact, dependency-free report."""
    validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=int)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence <= upper if index == n_bins - 1 else confidence < upper)
        if mask.any():
            rows.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float((predictions[mask] == labels[mask]).mean()),
                }
            )
    return rows


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    bins = reliability_bins(labels, probabilities, n_bins=n_bins)
    total = sum(row["count"] for row in bins)
    if total == 0:
        raise ValueError("At least one label and probability row is required")
    return float(sum(row["count"] * abs(row["accuracy"] - row["confidence"]) for row in bins) / total)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a fixed temperature to probabilities in log space."""
    validate_probabilities(probabilities)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def probability_histograms(probabilities: np.ndarray, n_bins: int = 10) -> dict[str, list[int]]:
    """Return per-class probability histograms for a compact report."""
    validate_probabilities(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return {
        f"class_{class_id}": np.histogram(probabilities[:, class_id], bins=edges)[0].astype(int).tolist()
        for class_id in range(probabilities.shape[1])
    }


def write_reliability_svg(rows: list[dict], path: str | Path, *, title: str = "Demo reliability diagram") -> None:
    """Write a small self-contained reliability diagram without matplotlib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = []
    for row in rows:
        x = 60 + 280 * row["confidence"]
        y = 340 - 280 * row["accuracy"]
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>')
    title_line = f'<text x="210" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>'
    accuracy_line = (
        '<text x="16" y="200" transform="rotate(-90 16 200)" '
        'text-anchor="middle" font-family="sans-serif">accuracy</text>'
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420" viewBox="0 0 420 420">
<rect width="100%" height="100%" fill="white"/>{title_line}
<line x1="60" y1="340" x2="340" y2="340" stroke="#222"/><line x1="60" y1="340" x2="60" y2="60" stroke="#222"/>
<line x1="60" y1="340" x2="340" y2="60" stroke="#9ca3af" stroke-dasharray="5 5"/>{"".join(points)}
<text x="210" y="385" text-anchor="middle" font-family="sans-serif">mean confidence</text>{accuracy_line}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_probability_histogram_svg(
    histograms: dict[str, list[int]], path: str | Path, *, title: str = "Demo probability histograms"
) -> None:
    """Write a self-contained class-probability histogram."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = max(max(values, default=0) for values in histograms.values()) or 1
    bars = []
    colors = ("#2563eb", "#dc2626", "#16a34a")
    for class_index, (name, values) in enumerate(histograms.items()):
        for bin_index, value in enumerate(values):
            x = 55 + bin_index * 28 + class_index * 8
            height = 250 * value / maximum
            y = 320 - height
            bars.append(f'<rect x="{x}" y="{y:.1f}" width="7" height="{height:.1f}" fill="{colors[class_index]}"/>')
    title_line = f'<text x="210" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>'
    bars_line = "".join(bars)
    axis_line = '<line x1="55" y1="320" x2="350" y2="320" stroke="#222"/>'
    axis_line += '<line x1="55" y1="320" x2="55" y2="60" stroke="#222"/>'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420" viewBox="0 0 420 420">
<rect width="100%" height="100%" fill="white"/>{title_line}
{axis_line}{bars_line}
<text x="205" y="365" text-anchor="middle" font-family="sans-serif">predicted probability bins</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")
