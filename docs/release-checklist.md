# Release checklist

Before making the repository public or pushing a release:

1. Run `python scripts/audit_public_repo.py`.
2. Run `python -m pytest -q` and `python -m compileall -q src tests`.
3. Confirm that all reported scores have scope and provenance in
   `experiments/results.csv` and `docs/kaggle-submission-audit.md`.
4. Confirm that no raw data, credentials, checkpoints, or generated submissions
   are staged.
5. Review `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `CITATION.cff`.
6. Review the final diff before committing or pushing.

The audit is intentionally conservative. If it reports a false positive, fix
the repository content or narrow the scanner with an explicit documented rule;
do not simply remove the check.
