# Kaggle workflow

The competition requires a notebook-based submission with a file named
`submission.csv`; the competition evaluates probability predictions with log
loss. The original project used Kaggle notebook packages for ModernBERT and
Gemma inference, but private dataset/model identifiers and credentials are not
included in this public repository.

Use the notebooks in `notebooks/` as templates after attaching your own permitted
data and model artifacts. Never commit `kaggle.json`, private dataset packages,
adapter weights, or generated submission files.
