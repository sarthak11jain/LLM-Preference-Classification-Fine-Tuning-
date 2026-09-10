# Reproducibility

The workflows keep data preparation, label mapping, fold construction, model
configuration, inference, and submission formatting explicit.

| Component | Implementation |
|---|---|
| Three-way preference ranking | `src/preference_classifier/data.py` |
| Grouped fold construction | `src/preference_classifier/splits.py` |
| Response-order augmentation | `src/preference_classifier/augmentation.py` |
| Swap-aware inference | `src/preference_classifier/inference.py` |
| Fold probability averaging | `src/preference_classifier/evaluation.py` |
| LoRA model construction | `src/preference_classifier/models.py` |
| Gemma-2 QLoRA configuration | `configs/gemma2_qlora_id_mod5.yaml` |
| Recorded metrics | `metrics/` and `experiments/results.csv` |

Run outputs can vary with data revisions, model revisions, random seeds, GPU
hardware, and library versions. Competition data, gated weights, and adapters
are supplied at runtime rather than stored in the repository.
