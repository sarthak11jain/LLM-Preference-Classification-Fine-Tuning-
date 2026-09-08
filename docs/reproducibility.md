# Reproducibility and provenance

The source project recorded the following implementation settings for the CV
claims represented here:

| Claim component | Evidence in this repository |
|---|---|
| Three-way preference ranking | `src/preference_classifier/data.py`, competition schema |
| Grouped cross-validation | `src/preference_classifier/splits.py`, `configs/*.yaml` |
| Response-order augmentation | `src/preference_classifier/augmentation.py` |
| Swap-aware TTA | `src/preference_classifier/augmentation.py` |
| Fold probability averaging | `src/preference_classifier/evaluation.py` |
| LoRA construction | `src/preference_classifier/models.py` |
| Gemma 2,048-token fold-1 metric | `evidence/gemma2_fold1_metrics.json` |
| ModernBERT 2,048-token fold-1 metric | `evidence/modernbert_fold1_metrics.json` |
| ModernBERT selected fold-2/fold-3 OOF metrics | `evidence/modernbert_folds23_metrics.json` |
| ModernBERT three-fold inference and public score | `evidence/modernbert_3fold_inference_metrics.json`, `experiments/kaggle-scores.csv` |
| Original full training workflows | `notebooks/` |

The recorded Gemma run used a 2,048-token budget, grouped folds, label
smoothing, LoRA, response swapping, and swap-aware validation inference. The
original artifact reports `0.9965459016988565` for fold 1. The Kaggle CLI
retrieved the matching ModernBERT fold-1 artifact, which reports
`1.0121814648626144`. The CLI competition history also provides public scores
`1.07855` for the frozen-head baseline, `1.01474` for the fold-1 inference
submission, and `1.01414` for the three-fold inference submission.

Exact reproduction additionally requires the competition data version, model
revision, seed, GPU/runtime, and attached adapter artifacts. Those large/private
artifacts are intentionally not committed.
