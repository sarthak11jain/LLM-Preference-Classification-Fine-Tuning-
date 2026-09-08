# LLM Preference Classification Fine-Tuning

Parameter-efficient fine-tuning for predicting which of two chatbot responses a human will prefer.

This repository is the public, reproducible companion to my Kaggle work on [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning). Each example contains a prompt and two candidate responses; the model predicts a three-class probability distribution:

```text
winner_model_a | winner_model_b | winner_tie
```

The project focuses on the engineering needed for preference probabilities rather than a hard winner: order-aware augmentation, prompt-grouped cross-validation, LoRA fine-tuning, label smoothing, swap-aware test-time augmentation, and fold probability averaging. Gemma-2 9B and ModernBERT-large are both first-class modeling workflows in this project.

## Verified project results

| Experiment | Log loss | Evaluation scope | Status |
|---|---:|---|---|
| ModernBERT frozen-head baseline | 1.07855 | Kaggle public leaderboard | Verified from CLI submission history |
| ModernBERT-large LoRA, fold 1 | 1.01218 | Local validation, fold 1 | Verified from Kaggle CLI-pulled metrics |
| ModernBERT-large LoRA, fold-1 inference | 1.01474 | Kaggle public leaderboard | Verified from CLI submission history |
| ModernBERT-large LoRA, three-fold inference | 1.01414 | Kaggle public leaderboard | Verified from CLI submission history |
| Gemma-2 9B LoRA | 0.99655 | Local validation, fold 1, 2,048-token input | Verified from `evidence/gemma2_fold1_metrics.json` |

The two local values are validation results, not final Kaggle leaderboard scores. The Gemma value is a fold-1 result; it must not be described as a completed three-fold aggregate. The ModernBERT public scores are separate inference submissions and should not be conflated with the local fold metric. Results are intentionally labeled this way so the repository justifies the CV claims without overstating them.

## Method

```text
prompt + response A + response B
              │
      tokenize with a fixed budget
              │
  original + response-order-swapped rows
              │
  GroupKFold by prompt (no prompt leakage)
              │
     ModernBERT-large or Gemma-2 9B
              │
          LoRA adapters
              │
        3-class probabilities
              │
  swap back → average TTA probabilities
              │
          average fold outputs
```

Training swaps the two responses and swaps the A/B labels at the same time. At inference, the swapped prediction is mapped back with `[1, 0, 2]` before averaging, so class semantics remain correct while positional bias is reduced.

## Repository layout

```text
src/preference_classifier/   reusable data, augmentation, evaluation, model, and ensemble code
configs/                     transparent ModernBERT and Gemma experiment settings
notebooks/                   selected source notebooks and readable experiment records
experiments/                 metric table and provenance notes
evidence/                    small, non-sensitive metric artifacts supporting reported values
docs/                        architecture, data, reproducibility, and Kaggle notes
tests/                       deterministic unit and smoke tests
```

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

The public repository does not distribute competition data, credentials, model weights, or adapter checkpoints. Obtain the data through Kaggle under its applicable terms, then point commands at a local directory containing `train.csv`, `test.csv`, and (for training) the three target columns. The expected schema is documented in [`docs/data-card.md`](docs/data-card.md).

Run the lightweight deterministic utilities without a model:

```powershell
python -m preference_classifier.cli validate-data --path path\to\train.csv
python -m preference_classifier.cli demo-swap
```

The public workflow also exposes configuration inspection and a deterministic
constant-probability submission:

```powershell
python -m preference_classifier.cli show-config --path configs\gemma2_lora.example.yaml
python -m preference_classifier.cli make-constant-submission --path path\to\test.csv --output outputs\constant_submission.csv
```

Full LoRA training requires a compatible CUDA GPU and the model files. The configurations in `configs/` are templates; they do not download or commit model weights.

## Reproduction tracks

- **Baseline:** constant probabilities and TF-IDF logistic regression for sanity checks.
- **Gemma-2 9B LoRA:** primary decoder fine-tuning path; the original run used a 2,048-token budget and recorded the fold-1 metric above.
- **ModernBERT-large LoRA:** primary encoder fine-tuning path with grouped folds, label smoothing, response swapping, and swap TTA.
- **Kaggle inference:** use the selected notebooks after attaching the permitted competition data, base model, and private adapter artifacts.

The notebooks are evidence and runnable templates; the reusable package is the canonical implementation boundary for new work.

## Limitations and honest scope

- Kaggle competition data and trained weights are not redistributed here.
- The best Gemma result is a single recorded validation fold, not a completed aggregate.
- The public leaderboard score and local validation scores use different evaluation scopes.
- Large-model training is compute-intensive and requires external GPU infrastructure.
- The repository documents the original implementation and provenance; it does not claim that a fresh run will produce identical scores without the same data, model revision, seed, hardware, and configuration.

## Sources and attribution

- Kaggle competition: [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)
- ModernBERT model family: [Answer.AI ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large)
- Gemma model family: [Google Gemma](https://huggingface.co/google/gemma-2-9b)
- Parameter-efficient fine-tuning: [Hugging Face PEFT](https://huggingface.co/docs/peft)

See [`docs/reproducibility.md`](docs/reproducibility.md) and [`experiments/results.csv`](experiments/results.csv) for provenance and configuration details.
The end-to-end public execution guide is [`docs/public-workflow.md`](docs/public-workflow.md).
Repository verification and publication boundaries are documented in [`docs/phase-5-quality.md`](docs/phase-5-quality.md).
Release and contribution guidance is available in [`docs/release-checklist.md`](docs/release-checklist.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and [`SECURITY.md`](SECURITY.md).
