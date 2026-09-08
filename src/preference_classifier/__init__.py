"""Reusable components for three-way LLM preference classification."""

LABEL_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")
INPUT_COLUMNS = ("prompt", "response_a", "response_b")

__all__ = ["INPUT_COLUMNS", "LABEL_COLUMNS"]
