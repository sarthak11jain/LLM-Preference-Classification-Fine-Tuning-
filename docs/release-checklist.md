# Release checklist

Before pushing a release:

1. Run `python -m pytest -q`.
2. Run `python -m ruff check src tests scripts`.
3. Run `python -m ruff format --check src tests scripts`.
4. Run `python -m compileall -q src scripts`.
5. Run `python scripts/audit_public_repo.py`.
6. Validate that results include their evaluation scope and configuration.
7. Confirm that raw data, credentials, model weights, adapters, and generated
   submissions are not staged.
8. Review the README and final diff.
