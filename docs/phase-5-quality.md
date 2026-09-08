# Phase 5 — public repository quality

The repository is intended to be inspectable and safe to publish. This phase
adds deterministic API coverage and protects the evidence files that support
the reported CV results from the broad CSV ignore rule.

## Local verification

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src tests
python -m preference_classifier.cli demo-swap
```

The GitHub Actions workflow runs the test suite and compilation checks on every
push and pull request. Tests cover grouped prompt folds, schema validation,
probability metrics, submission formatting, out-of-fold completeness, swap
augmentation, configuration loading, and the deterministic baseline helpers.

## Publication boundary

The public repository deliberately excludes competition data, model weights,
adapter checkpoints, credentials, generated outputs, and local environments.
Metric CSVs under `experiments/` and `evidence/` are explicitly allow-listed
because they are small provenance artifacts rather than raw data.

## Contribution rule

Any new reported score must include its evaluation scope, model/configuration,
source artifact, and an honest reproducibility caveat. New code should remain
path-independent and must not require a local Kaggle credential to import or
test.
