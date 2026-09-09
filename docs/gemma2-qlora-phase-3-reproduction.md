# Gemma-2 QLoRA Phase 3 reproduction

Phase 3 submits the cleaned full-data training notebook as an executable Kaggle
kernel. It uses the individual project configuration audited in Phase 1 and
the evaluation rule `id % 5 == 0`.

## Submitted run

| Field | Value |
|---|---|
| Kaggle kernel | [`sarthak11j/gemma-2-9b-qlora-id-mod-5-evaluation`](https://www.kaggle.com/code/sarthak11j/gemma-2-9b-qlora-id-mod-5-evaluation) |
| Kernel version | 4 |
| Status at submission | Running after `pyarrow_hotfix` compatibility retry |
| Training rows | 46,001 |
| Evaluation rows | 11,476 |
| Training length | 1,024 tokens |
| Model | Gemma-2 9B 4-bit |
| Adapter method | PEFT QLoRA |
| Epochs | 1 |
| Evaluation rule | `id % 5 == 0` |

The package source is maintained in
`kaggle/submissions/gemma2_qlora_id_mod5_training/`; the readable notebook is
`notebooks/gemma2_qlora_id_mod5_training.ipynb`.

## Result recording rule

No new score is added to `experiments/results.csv` until the Kaggle run
finishes and its output contains machine-readable metrics. The completed run
must provide:

- `metrics.json` with evaluation log loss and row counts;
- the saved adapter/tokenizer output;
- the final kernel status; and
- a submission record if inference is subsequently executed.

The historical `0.9371` / approximately `0.941` values remain documented in
the Phase 1 audit as the target result for this reconstruction. They are not
duplicated as a newly verified run until this execution produces its own
artifacts.

## Version 1 execution note

Versions 1 and 2 stopped during imports because package installation left the
Kaggle runtime with a binary NumPy/Pandas mismatch. Version 3 preserved that
numeric stack but stopped because `datasets` could not import the missing
`pyarrow_hotfix` module. Version 4 adds that small compatibility package with
`--no-deps`. No model or evaluation conclusion was drawn from the failed
versions.
