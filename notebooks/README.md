# Notebooks

The notebooks are thin runtime entrypoints. Reusable data, model,
training, inference, and evaluation logic lives in
`src/preference_classifier`.

- `gemma2_qlora_id_mod5_training.ipynb` — primary Gemma-2 9B 4-bit QLoRA run.
- `gemma2_qlora_id_mod5_inference.ipynb` — adapter inference and submission.
- `modernbert_lora_3fold_inference_submission.ipynb` — grouped-fold
  ModernBERT comparison workflow.

They require competition data, model weights, and a compatible GPU at runtime;
the CPU-safe path is documented in the root README.
