# Gemma-2 QLoRA Phase 1 audit

This document records the reconstruction of the Gemma-2 9B 4-bit QLoRA
workflow used for this project. The notebooks were retrieved with the Kaggle
CLI and checked against the locally available competition files. The public
repository contains a cleaned, adapted implementation; it does not redistribute
the downloaded model weights.

## Identified notebooks

| Workflow | Kaggle kernel | Status | GPU | Internet |
|---|---|---|---|---|
| Training | `duohanwang/training-gemma-2-9b-4-bit-qlora-fine-tunin-25a55e` | Complete | Required | Enabled |
| Inference | `duohanwang/inference-gemma-2-9b-4-bit-qlora-6b251d` | Complete | Required | Disabled |

Both kernels are public Kaggle notebooks associated with the LLM Classification
Finetuning competition. The retrieved source notebooks contain zero executed
cells and therefore do not contain serialized metric objects; the reported
metrics are stated in their markdown result sections. The Kaggle CLI did,
however, expose a checkpoint output from training and a three-row inference
submission, whose hashes are recorded in
`evidence/gemma2_qlora_public_notebook_run.json`. The downloaded training
checkpoint reports `max_steps=20`, matching the notebook's 100-row demo path;
it is not the full-data adapter behind the reported metrics.

## Dataset size and split

The locally available competition training file contains **57,477 rows** and
the expected nine columns: `id`, `model_a`, `model_b`, `prompt`, `response_a`,
`response_b`, and the three winner columns. The test file contains three rows.

The training notebook states that evaluation uses `id % 5 == 0`. On the
available 57,477-row training file, that produces:

| Partition | Rule | Rows |
|---|---|---:|
| Training | `id % 5 != 0` | 46,001 |
| Evaluation | `id % 5 == 0` | 11,476 |

All training IDs are unique in the inspected file.

## Training configuration found

| Setting | Value |
|---|---|
| Base model | `unsloth/gemma-2-9b-it-bnb-4bit` |
| Model family | Gemma-2 9B Instruct |
| Quantization | 4-bit |
| Fine-tuning | PEFT QLoRA |
| Training epochs | 1 |
| Training max length | 1,024 tokens |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj` |
| Optimizer | `adamw_8bit` |
| Learning rate | `2e-4` |
| Precision | fp16 |
| Gradient accumulation | 2 |
| Evaluation split | `id % 5 == 0` |

The visible notebook source includes a first-100-row demonstration selector.
That selector must be removed or made explicitly optional in the production
reproduction notebook. The reported `0.9371` evaluation value cannot be
recreated from that demonstration selector alone without confirming the
executed full-data state.

## Inference configuration found

| Setting | Value |
|---|---|
| Base model | Gemma-2 9B Instruct 4-bit artifact |
| Maximum length | 2,048 tokens |
| Batch size | 8 |
| Device layout | Two CUDA devices |
| Adapter | Unmerged LoRA adapter |
| TTA default | Disabled in the retrieved notebook |
| Output | `submission.csv` with three probabilities per ID |

The inference source contains optional response-order TTA. When enabled, the
swapped output exchanges the A/B probability columns before blending with the
original output. The notebook uses a weighted blend of `0.45` original and
`0.55` swapped predictions.

## Reported results

| Evaluation | Log loss | Result status |
|---|---:|---|
| Evaluation split | 0.9371 | Reported in the training notebook markdown |
| Public leaderboard | approximately 0.941 | Reported in the training/inference notebook markdown |

These values are treated as results of this individual project workflow and are
included in the main results table with `reported` status. They are not labeled
as a fresh machine-verified rerun. The downloaded adapter and submission provide
artifact provenance, but the source notebooks do not serialize the metric
calculation itself.

## Phase 1 conclusion

The workflow, data size, split rule, model settings, and reported metrics are
identified. The clean production notebooks and reusable implementation preserve
the `id % 5 == 0` evaluation split, while the public result record is tracked
separately from fresh machine-verified runs.
