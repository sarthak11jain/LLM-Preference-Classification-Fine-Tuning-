# Notebooks

The notebooks contain the complete GPU workflows:

- `gemma2_qlora_id_mod5_training.ipynb` — Gemma-2 9B 4-bit QLoRA training with
  the `id % 5 == 0` evaluation split.
- `gemma2_qlora_id_mod5_inference.ipynb` — 2,048-token inference and probability
  submission generation.
- `gemma2_lora_training.ipynb` — Gemma-2 9B LoRA training with grouped folds,
  response swapping, and probability evaluation.
- `modernbert_lora_offline_full_submission.ipynb` — ModernBERT-large LoRA
  training and fold evaluation.
- `modernbert_lora_3fold_inference_submission.ipynb` — fold loading, swapped
  inference, probability remapping, and fold averaging.

The notebooks expect competition data and model artifacts to be attached at
runtime. The portable utilities live in `src/preference_classifier/`.
