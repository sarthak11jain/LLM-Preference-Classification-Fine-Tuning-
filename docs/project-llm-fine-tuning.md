# LLM Fine-Tuning Project

## Purpose

This repository contains a package-first implementation of three-way LLM
preference classification. Given a prompt and two chatbot responses, the
system predicts whether response A wins, response B wins, or the responses
are tied.

The competition metric is multiclass log loss, so the workflows produce
probabilities rather than only hard class predictions.

## Primary methodology

- Gemma-2 9B Instruct with 4-bit PEFT QLoRA.
- LoRA-based ModernBERT-large comparison workflow.
- Response-order augmentation with A/B label remapping.
- Swap-aware test-time augmentation and probability averaging.
- Prompt-grouped cross-validation for the leakage-aware comparison track.
- Deterministic `id % 5 == 0` validation for the primary Gemma result.
- 1,024-token training and 2,048-token inference budgets for the primary run.

## Results policy

The primary recorded result is Gemma-2 9B 4-bit QLoRA with log loss `0.9371`
on the deterministic `id % 5 == 0` validation split. The recorded public
leaderboard result is `0.941`.

Results from deterministic ID validation, grouped local validation, and public
leaderboard evaluation are kept in separate tables. Scores are not treated as
directly comparable when their protocols differ.

The grouped full-data aggregate is not recorded until a real GPU run produces
it. The ablation table remains an empty schema until controlled experiments are
completed; no values are inferred from unrelated runs.

## Repository decisions

- The repository name remains `LLM-Preference-Classification-Fine-Tuning-`.
- `src/preference_classifier/` is the source of truth for reusable logic.
- Notebooks are thin runtime entrypoints for GPU workflows.
- CPU verification uses synthetic data under `data/demo/`.
- The constant baseline uses a uniform `[1/3, 1/3, 1/3]` prior for unlabeled
  test data.
- Calibration tooling reports reliability bins, ECE, Brier score, probability
  histograms, and optional swap-TTA comparisons.
- README and documentation use project-first language and do not contain
  private provenance or CV-justification material.

## Important paths

```text
src/preference_classifier/       reusable package and CLI implementation
configs/                         ID-split, grouped, demo, and model configs
data/demo/                       synthetic CPU-safe input files
experiments/                     separated local, leaderboard, and ablation records
metrics/                         compact records for completed experiments
notebooks/                       thin Gemma and ModernBERT runtime entrypoints
reports/demo/                    committed CPU demo outputs and visualizations
tests/                           unit, CLI, calibration, and smoke tests
```

## Public commands

```powershell
python -m preference_classifier.cli demo --output-dir reports/demo
python -m preference_classifier.cli show-config --path configs\gemma2_qlora_id_mod5.yaml
python -m preference_classifier.cli validate-data --path path\to\train.csv
python -m preference_classifier.cli make-constant-submission --path path\to\test.csv --output outputs\constant.csv
python -m preference_classifier.cli train --config configs\demo_cpu.yaml --train data\demo\train.csv --output-dir outputs\tfidf --backend tfidf
python -m preference_classifier.cli evaluate --labels data\demo\validation.csv --predictions data\demo\validation_predictions.csv
python -m preference_classifier.cli calibrate --labels data\demo\validation.csv --predictions data\demo\validation_predictions.csv --output-dir outputs\calibration
```

Transformer training and inference require the pinned GPU dependencies, model
weights, permitted competition data, and a compatible GPU runtime. Those
artifacts and credentials are intentionally not stored in this repository.

## Verification status

The latest repository update passed:

- 16 automated tests;
- Ruff linting and formatting checks;
- approximately 66% package coverage with a 50% CI floor;
- Python compilation checks;
- notebook JSON parsing;
- the full public-repository audit;
- the CPU demo workflow.

The repository history contains the implementation upgrade commit
`2d1db93` (`Upgrade preference classification showcase workflow`).

## Remaining work

1. Run the grouped full-data workflow on a compatible GPU.
2. Add only measured grouped aggregate and controlled ablation results.
3. Generate real OOF probability artifacts if model calibration is to be
   discussed beyond the CPU demonstration.
4. Re-run CI after future repository changes.
