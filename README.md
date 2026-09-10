# LLM Preference Classification Fine-Tuning

[![Quality](https://github.com/sarthak11jain/LLM-Preference-Classification-Fine-Tuning-/actions/workflows/quality.yml/badge.svg)](https://github.com/sarthak11jain/LLM-Preference-Classification-Fine-Tuning-/actions/workflows/quality.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/llm-classification-finetuning)

Fine-tuning workflows for predicting which of two chatbot responses people
prefer. The project turns paired conversations into calibrated probabilities for
three outcomes: response A wins, response B wins, or the annotators tie.

## Problem

Each example contains a prompt and two candidate responses. The model produces
probabilities in the competition submission format:

```text
winner_model_a | winner_model_b | winner_tie
```

The evaluation metric is multiclass log loss, which rewards accurate and well-
calibrated probabilities rather than only the correct hard label.

## Results

| Model | Evaluation scope | Input length | Log loss |
|---|---|---:|---:|
| Gemma-2 9B 4-bit QLoRA | `id % 5 == 0` validation split | 1,024 train / 2,048 inference | **0.9371** |
| Gemma-2 9B 4-bit QLoRA | Public leaderboard | 2,048 | **~0.941** |
| Gemma-2 9B LoRA | Fold 1 validation | 2,048 | 0.99655 |
| ModernBERT-large LoRA | Fold 1 validation | 2,048 | 1.01218 |
| ModernBERT-large LoRA | Three-fold leaderboard submission | 2,048 | 1.01414 |
| ModernBERT frozen classifier head | Public leaderboard | 2,048 | 1.07855 |
| Constant probability baseline | Public leaderboard | — | 1.09794 |

## Approach

### Data and labels

The training data contains paired chatbot responses and one-hot preference
targets. Labels are mapped to three classes and probabilities are validated
before a submission is written.

### Model training

The primary workflow uses Gemma-2 9B Instruct with 4-bit quantization and PEFT
QLoRA. It trains for one epoch with a 1,024-token training budget and evaluates
using the deterministic `id % 5 == 0` split. ModernBERT-large provides a strong
encoder comparison using the same three-way objective.

### Generalization

- Prompts define groups for fold construction, preventing repeated prompts from
  crossing a train/evaluation boundary.
- Response-order augmentation trains on both A/B presentations and swaps the
  corresponding labels.
- Swap-aware inference remaps the A/B probabilities before averaging.
- Fold predictions are combined by probability averaging rather than majority
  voting.

### Insights

The strongest improvement comes from using a long-context decoder with
parameter-efficient adaptation while preserving probability calibration. The
response swap path directly addresses position bias, and grouped folds make the
validation estimate more representative when prompts repeat.

## Repository structure

```text
.
├── configs/                 model and experiment configurations
├── docs/                    architecture, data, workflows, and references
├── metrics/                 compact metric snapshots
├── experiments/             result tables and experiment notes
├── notebooks/               training and inference workflows
├── scripts/                 publication and quality checks
├── src/preference_classifier/
│   ├── baselines.py         constant and TF-IDF reference models
│   ├── data.py              schema and label handling
│   ├── augmentation.py      response swaps and label remapping
│   ├── preprocessing.py     formatting and token-budget helpers
│   ├── splits.py            grouped fold construction
│   ├── models.py            lazy LoRA model construction
│   ├── training.py          loss, seeding, and training helpers
│   ├── inference.py         TTA and fold ensembling
│   ├── evaluation.py        log loss and probability validation
│   └── submission.py        submission formatting
├── tests/                   deterministic unit and API tests
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

CPU-safe commands:

```powershell
python -m preference_classifier.cli demo-swap
python -m preference_classifier.cli validate-data --path path\to\train.csv
python -m preference_classifier.cli show-config --path configs\gemma2_qlora_id_mod5.yaml
python -m preference_classifier.cli make-constant-submission --path path\to\test.csv --output outputs\constant_submission.csv
```

## Workflows

Use [`configs/gemma2_qlora_id_mod5.yaml`](configs/gemma2_qlora_id_mod5.yaml),
[`notebooks/gemma2_qlora_id_mod5_training.ipynb`](notebooks/gemma2_qlora_id_mod5_training.ipynb),
and [`notebooks/gemma2_qlora_id_mod5_inference.ipynb`](notebooks/gemma2_qlora_id_mod5_inference.ipynb)
for the primary Gemma-2 workflow.

The ModernBERT workflows are available in
[`notebooks/modernbert_lora_offline_full_submission.ipynb`](notebooks/modernbert_lora_offline_full_submission.ipynb)
and [`notebooks/modernbert_lora_3fold_inference_submission.ipynb`](notebooks/modernbert_lora_3fold_inference_submission.ipynb).

Competition data, gated model weights, and trained adapters are supplied at
runtime. They are not distributed in this repository. See
[`docs/data-card.md`](docs/data-card.md), [`docs/public-workflow.md`](docs/public-workflow.md),
and [`docs/kaggle.md`](docs/kaggle.md) for execution details.

## Reproducibility

The repository records the model configuration, split rule, token budgets, and
result values used by each workflow. Exact scores can vary with data revisions,
model revisions, random seeds, GPU hardware, and library versions. The compact
metric files under [`metrics/`](metrics/) provide the corresponding run details.

## References

- [Kaggle LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)
- [Answer.AI ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)
- [Google Gemma-2 9B](https://huggingface.co/google/gemma-2-9b)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and
[`docs/release-checklist.md`](docs/release-checklist.md) before publishing
changes.
