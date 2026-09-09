# Gemma-2 QLoRA Phase 4 inference

Phase 4 prepares the inference and submission stage for the trained adapter.
The canonical notebook is `notebooks/gemma2_qlora_id_mod5_inference.ipynb` and
the Kaggle package is under
`kaggle/submissions/gemma2_qlora_id_mod5_inference/`.

## Execution contract

The inference run must receive:

- the competition `test.csv`;
- the compatible Gemma-2 9B 4-bit base model;
- the adapter produced by the validated Phase 3 training run; and
- a GPU runtime with the pinned inference libraries.

It uses a 2,048-token maximum sequence length and writes one row per test ID
with `winner_model_a`, `winner_model_b`, and `winner_tie` probabilities. Optional
response-order TTA remaps the swapped A/B outputs before a weighted blend.

The inference package is deliberately not submitted until the training run
produces and validates its adapter and `metrics.json` output.
