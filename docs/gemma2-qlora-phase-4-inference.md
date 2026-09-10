# Gemma-2 QLoRA Phase 4 inference

Phase 4 records the inference and submission stage for the downloaded public
notebook artifact.
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

The public inference notebook produced a valid three-row `submission.csv` using
its mounted adapter source. This adapter source is separate from the downloaded
training demo checkpoint.
Its columns, row count, and SHA-256 are recorded in
`evidence/gemma2_qlora_public_notebook_run.json`. The repository does not claim
that this local checkout freshly reproduced the approximately `0.941` public
score; that value is the notebook's reported leaderboard result.
