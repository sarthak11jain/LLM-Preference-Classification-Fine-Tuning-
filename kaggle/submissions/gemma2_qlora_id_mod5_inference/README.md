# Gemma-2 QLoRA inference package

This Kaggle package runs the canonical inference notebook with a user-attached
Gemma-2 base model and trained LoRA adapter. Configure `LLM_DATA_DIR`,
`LLM_BASE_MODEL`, and `LLM_ADAPTER_DIR` for the mounted Kaggle inputs before
execution.

The package is prepared but should be submitted only after the Phase 3 training
run produces a validated adapter artifact.
