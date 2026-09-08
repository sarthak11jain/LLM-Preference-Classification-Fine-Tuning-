# Phase 3 — source migration map

Phase 3 turns the original private project into a reviewable public package. The public package contains reusable, path-independent behavior; GPU execution notebooks preserve the evidence trail for the two primary workflows.

| Original source | Public home | Decision |
|---|---|---|
| `evaluate_local_tfidf_lr.py` | `src/preference_classifier/baselines.py` | Migrated as a lightweight reference baseline |
| `baseline_submission.py`, `generate_submission_baseline.py` | `baselines.py`, `submission.py` | Consolidated and sanitized |
| `generate_submission_modernbert*.py` | `config.py`, `data.py`, `augmentation.py`, `preprocessing.py`, `models.py`, `training.py`, `evaluation.py`, `submission.py` | Reusable behavior migrated; full GPU run remains in evidence notebook |
| `generate_submission_gemma2_lora*.py` | Same package modules | Reusable behavior migrated; full GPU run remains in evidence notebook |
| `train_vertex_lora.py` | `docs/reproducibility.md` and evidence notebooks | Environment-specific orchestration intentionally excluded |
| `inspect_data.py`, `download_data.py` | `docs/` | Dataset and acquisition context documented; no credential logic copied |
| `kaggle_notebook_submit.py`, `poll_submission.py`, helper scripts | `docs/kaggle-submission-audit.md` | Operational/private tooling excluded from the public package |
| `setup.py`, requirements files | `pyproject.toml` | Consolidated into one public package definition |

Excluded from the public repository: credentials, raw competition data, checkpoints/adapters, private Kaggle package paths, and machine-specific absolute paths.
