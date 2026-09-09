# Notebooks

These notebooks are selected public evidence from the original project. They
show the complete GPU workflows that motivated the reusable package:

- `gemma2_qlora_id_mod5_training.ipynb` — clean full-data Gemma-2 9B 4-bit
  QLoRA training with the project evaluation rule `id % 5 == 0`.
- `gemma2_qlora_id_mod5_inference.ipynb` — 2,048-token adapter inference and
  probability submission generation.

The executable Kaggle package for the training reproduction is under
`kaggle/submissions/gemma2_qlora_id_mod5_training/`.

The matching inference package is prepared under
`kaggle/submissions/gemma2_qlora_id_mod5_inference/` and awaits a validated
trained adapter.

- `gemma2_lora_external_gpu_training.ipynb` — Gemma-2 9B LoRA, grouped folds,
  label smoothing, response swapping, swap TTA, and OOF evaluation.
- `modernbert_lora_offline_full_submission.ipynb` — ModernBERT-large LoRA
  training and fold evaluation.
- `modernbert_lora_3fold_inference_submission.ipynb` — loading fold adapters,
  original/swapped inference, probability remapping, and fold averaging.

The notebooks expect private/local Kaggle inputs and GPU model artifacts. They
are not the lightweight quickstart. Use `src/preference_classifier/` for
portable schema, augmentation, evaluation, and model-construction utilities.
