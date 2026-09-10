# Kaggle workflow

The competition expects a notebook to write `submission.csv` with one row per
test ID and three probability columns. The competition metric is multiclass log
loss.

Use the thin notebooks in `notebooks/` or the `train` and `predict` CLI
commands after attaching permitted competition data and model artifacts to a
GPU runtime. Keep credentials, private datasets, model weights, adapters, and
generated submissions outside this repository.

The Gemma-2 workflow uses 4-bit QLoRA and may require gated model access,
bitsandbytes support, and substantial GPU memory. The synthetic CPU demo is the
supported way to verify the package without those resources.
