# Public workflow

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install -e .
```

## Check the package

```powershell
preference-classifier demo-swap
preference-classifier validate-data --path path\to\train.csv
preference-classifier show-config --path configs\gemma2_qlora_id_mod5.yaml
preference-classifier demo --output-dir outputs\demo
```

## Generate a baseline

```powershell
preference-classifier make-constant-submission `
  --path path\to\test.csv `
  --output outputs\constant_submission.csv
```

## Run a model workflow

Attach the competition data, base model, and adapter to a GPU runtime, then open
the matching notebook under `notebooks/`. The reusable implementation in
`src/preference_classifier/` handles schema validation, preprocessing,
augmentation, evaluation, and submission formatting.

## Validate a change

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m compileall -q src scripts
python scripts/audit_public_repo.py
```
