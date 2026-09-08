# Public workflow

This is the shortest path from a clean checkout to a valid artifact. It is
deliberately split into a CPU-safe smoke path and GPU model paths.

## 1. Install and validate

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
preference-classifier demo-swap
preference-classifier validate-data --path path\to\train.csv
```

## 2. Inspect model configuration

```powershell
preference-classifier show-config --path configs\modernbert_lora.yaml
preference-classifier show-config --path configs\gemma2_lora.example.yaml
```

The YAML files document the two CV-facing model families. They are templates:
the repository does not download competition data, gated model weights, or
private adapters automatically.

## 3. Produce a deterministic baseline submission

The test file must contain `id`, `prompt`, `response_a`, and `response_b`.

```powershell
preference-classifier make-constant-submission `
  --path path\to\test.csv `
  --output outputs\constant_submission.csv
```

## 4. Run the model workflows

For GPU execution, attach the allowed competition data and model artifacts,
then use the corresponding notebook in `notebooks/`. The canonical reusable
components are in `src/preference_classifier/`: grouped splits, response
swapping, preprocessing, LoRA defaults, loss helpers, TTA, fold averaging, and
submission validation. This separation keeps the public code reviewable while
allowing Kaggle or another GPU host to supply the large/private artifacts.

## 5. Verify the reported evidence

The repository's reported values are evidence records, not promises that a
fresh run will reproduce the same score. See `experiments/results.csv`, the
JSON files under `evidence/`, and `docs/kaggle-submission-audit.md` for scope,
source, and limitations of each value.
