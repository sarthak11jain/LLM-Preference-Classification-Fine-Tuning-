# Experiment records

The CSV files use one row per measured run. They keep validation, leaderboard,
and ablation scopes separate and record the protocol needed to interpret each
score.

- `local_validation.csv` contains local validation results.
- `public_leaderboard.csv` contains competition leaderboard results.
- `ablation.csv` is an intentionally empty schema until controlled runs are
  completed. No ablation number is fabricated from incomparable experiments.

The grouped-validation pipeline is ready for a future GPU run. When that run
is completed, add its exact split, row count, seed, model revision, token
budgets, augmentation settings, score, and metric artifact here.
