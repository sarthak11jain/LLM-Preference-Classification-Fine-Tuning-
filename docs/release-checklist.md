# Release checklist

Before pushing a release:

1. Run `python -m pytest -q`.
2. Run `python -m compileall -q src scripts`.
3. Run `python scripts/audit_public_repo.py`.
4. Validate that results include their evaluation scope and configuration.
5. Confirm that raw data, credentials, model weights, adapters, and generated
   submissions are not staged.
6. Review the README and final diff.
