# CV claim provenance

This table maps each CV bullet to public evidence in the repository.

| CV claim | Repository evidence | Boundary |
|---|---|---|
| Engineered an LLM preference-ranking system for Kaggle, predicting three-way human choices from paired chatbot responses | `README.md`, `docs/data-card.md`, `src/preference_classifier/data.py`, Kaggle competition link | Project scope and schema are directly reproducible; competition data remains external |
| Fine-tuned Gemma-2 9B and ModernBERT-large with LoRA, grouped CV, label smoothing, and 2K inputs | `configs/`, `src/preference_classifier/models.py`, `src/preference_classifier/splits.py`, `src/preference_classifier/training.py`, selected notebooks, and CLI-pulled Kaggle notebooks | Full model runs require GPU/model artifacts; CLI metrics confirm ModernBERT fold 1 at 2,048 tokens and the Gemma artifact confirms the Gemma fold-1 configuration |
| Strengthened generalization with response-order augmentation, swap-aware TTA, fold ensembling, and probability averaging | `src/preference_classifier/augmentation.py`, `src/preference_classifier/evaluation.py`, inference notebooks, `docs/architecture.md` | The transformations and averaging logic are unit-tested; aggregate results are labeled by scope |
| Recorded 0.99655 validation log loss with Gemma-2 9B, outperforming the 1.01218 ModernBERT LoRA baseline on the same split | `evidence/gemma2_fold1_metrics.json`, `evidence/modernbert_fold1_metrics.json`, `experiments/results.csv` | Both are local fold-1 validation values, not leaderboard scores; CLI-pulled metrics support the comparison |

The separate `id % 5 == 0` Gemma-2 QLoRA reconstruction is audited in
`docs/gemma2-qlora-phase-1-audit.md` and
`evidence/gemma2_qlora_phase1_audit.json`. Its reported `0.9371` evaluation and
approximately `0.941` leaderboard values are intentionally kept outside the
main results table until the full-data workflow is reproduced with
machine-readable artifacts.

The repository deliberately does not turn a fold-1 local value into a final
leaderboard claim. This distinction is part of the evidence quality of the
project.
