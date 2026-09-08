# Contributing

Thank you for helping improve this project. Contributions should keep the
repository reproducible, honest about evaluation scope, and safe to publish.

Before opening a pull request:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src tests
```

Please do not add competition data, model weights, adapter checkpoints,
credentials, generated submissions, or machine-specific absolute paths.
When adding a result, include the model/configuration, split, metric, source
artifact, and whether the value is local validation or leaderboard evaluation.

Keep public imports path-independent and preserve the CPU-safe behavior of the
package wherever possible.
