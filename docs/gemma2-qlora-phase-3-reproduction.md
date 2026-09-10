# Gemma-2 QLoRA Phase 3 artifact adoption

Phase 3 adopts the public training notebook's recorded configuration and
downloadable checkpoint evidence without requiring a new full Kaggle run. It
uses the individual project configuration audited in Phase 1 and the evaluation
rule `id % 5 == 0`.

## Submitted run

| Field | Value |
|---|---|
| Kaggle kernel | [`sarthak11j/gemma-2-9b-qlora-id-mod-5-evaluation`](https://www.kaggle.com/code/sarthak11j/gemma-2-9b-qlora-id-mod-5-evaluation) |
| Kernel version | 5 (attempted validation only) |
| Status at submission | Error; not used as result evidence |
| Training rows | 46,001 |
| Evaluation rows | 11,476 |
| Training length | 1,024 tokens |
| Model | Gemma-2 9B 4-bit |
| Adapter method | PEFT QLoRA |
| Epochs | 1 |
| Evaluation rule | `id % 5 == 0` |

The cleaned package source is maintained in
`kaggle/submissions/gemma2_qlora_id_mod5_training/`; the readable notebook is
`notebooks/gemma2_qlora_id_mod5_training.ipynb`.

## Adopted result record

The public notebook reports evaluation log loss `0.9371` and public leaderboard
log loss approximately `0.941`. These values are recorded in
`experiments/results.csv` with `reported` status and are backed by
`evidence/gemma2_qlora_public_notebook_run.json`.

The CLI also downloaded a checkpoint and a valid inference submission. Their
hashes establish which public artifacts were inspected; they do not turn the
markdown-reported metrics into a fresh rerun.

No further Kaggle training run is required for the repository documentation
track. A future rerun may be used as an independent reproducibility check.

## Version 1 execution note

Versions 1 and 2 stopped during imports because package installation left the
Kaggle runtime with a binary NumPy/Pandas mismatch. Versions 3 and 4 isolated
the numerical stack but still failed while importing incompatible `datasets`
and PyArrow components. Version 5 removes that unnecessary dependency and uses
a native PyTorch dataset wrapper. No model or evaluation conclusion was drawn
from the failed versions.
