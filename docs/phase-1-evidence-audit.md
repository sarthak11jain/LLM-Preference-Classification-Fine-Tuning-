# Phase 1 — Evidence and scope audit

Audit target: supplied local LLM fine-tuning project workspace
Audit date: 2026-09-08
Public repository target: `LLM-Preference-Classification-Fine-Tuning-`

## Phase 1 outcome

The original project contains a substantive implementation, not only an idea:
15 Python source files, 7 working notebooks, 4 operational scripts, Kaggle
submission packages, a Gemma adapter artifact, raw competition data, and a
README containing experiment history and CV wording.

The implementation supports the technical story in the CV: three-way preference
prediction, ModernBERT and Gemma LoRA workflows, prompt-grouped folds, label
smoothing, response-order augmentation, swap-aware TTA, and fold probability
averaging.

The audit also found different evidence strengths for the reported metrics. The
public repository must preserve those distinctions.

## CV claim evidence matrix

| CV statement | Local implementation evidence | Metric/result evidence | Phase 1 status |
|---|---|---|---|
| Engineered an LLM preference-ranking system for Kaggle, predicting three-way human choices from paired chatbot responses | `src/generate_submission_modernbert.py`, `src/generate_submission_gemma2_lora.py`, the training/inference notebooks, competition schema, and the supplied README | Official Kaggle competition defines paired chatbot-response preference prediction and multiclass log loss | **Verified project scope** |
| Fine-tuned Gemma-2 9B and ModernBERT-large with LoRA | `src/generate_submission_gemma2_lora.py`, `src/generate_submission_modernbert_lora.py`, `src/train_vertex_lora.py`, and corresponding notebooks | Kaggle kernel metadata identifies the attached ModernBERT/Gemma model sources; Gemma adapter metadata identifies a PEFT adapter | **Verified implementation path** |
| Used grouped CV | `build_groups`, `GroupKFold`, and fold loops appear in the ModernBERT/Gemma workflows and notebooks | Configurations use 3 folds; exact aggregate output is not present for every run | **Verified method; aggregate evidence incomplete** |
| Used label smoothing | `CrossEntropyLoss(label_smoothing=...)` appears in the ModernBERT and Gemma workflows | Gemma notebook records `0.02`; ModernBERT workflows expose label-smoothing settings | **Verified implementation/configuration** |
| Used 2K inputs | Gemma training/inference notebooks and metrics artifact record `max_length=2048`; ModernBERT inference workflow requests 2048 | `kaggle/datasets/gemma2_fold1_adapter/fold_1/metrics.json` records 2048 | **Verified for Gemma fold-1; verify exact ModernBERT run** |
| Used response-order augmentation | Training code duplicates rows and swaps `response_a`/`response_b`; labels are swapped at the same time | Present in both model workflows and notebooks | **Verified implementation** |
| Used swap-aware TTA | Inference code predicts original and swapped rows, then applies the A/B permutation `[1, 0, 2]` before averaging | Present in ModernBERT and Gemma inference notebooks | **Verified implementation** |
| Used fold ensembling and probability averaging | Three-fold inference notebook averages fold outputs; Gemma/ModernBERT workflows accumulate fold probabilities | Final complete aggregate metric is not present for every model | **Verified workflow; aggregate result incomplete** |
| Recorded Gemma-2 9B validation log loss `0.99655` | Gemma external-GPU notebook and adapter artifact | `best_val_log_loss=0.9965459016988565`, fold 1, epoch 1, 2048 tokens, `use_4bit=false` | **Verified local fold-1 result** |
| Recorded ModernBERT LoRA validation log loss `1.01218` | ModernBERT LoRA workflow, Kaggle CLI-pulled `try-sj-12` notebook/log, and `try-sj-12/fold_1/metrics.json` | `best_val_log_loss=1.0121814648626144`, fold 1, max length 2048 | **Verified local fold-1 result** |
| Gemma-2 9B outperformed ModernBERT LoRA on the same split | Gemma fold-1 metrics artifact and ModernBERT `try-sj-12/fold_1/metrics.json` | Gemma `0.9965459016988565` versus ModernBERT `1.0121814648626144`; both are fold-1 local validation records | **Verified recorded comparison; confirm identical split metadata in final write-up** |
| Recorded ModernBERT frozen-head public score `1.07855` | Kaggle CLI competition submissions plus `try-sj-11` pulled notebook/output | CLI score `1.07855` is attached to the submission immediately following the `try-sj-11` ModernBERT head-baseline run | **Verified public score** |

## Metric artifacts found

### Gemma-2 9B

The strongest local evidence is:

```text
kaggle/datasets/gemma2_fold1_adapter/fold_1/metrics.json
```

It records:

- fold: `1`;
- best epoch: `1`;
- validation log loss: `0.9965459016988565`;
- max length: `2048`;
- 4-bit loading: `false`;
- runtime: `28,740` seconds;
- model path: a private Kaggle-mounted Gemma-2 9B model.

This supports the CV value as a local fold-1 validation result. It does not
support a completed three-fold aggregate or leaderboard claim.

### ModernBERT LoRA

The Kaggle CLI pulled the original `try-sj-12` output. Its `fold_1/metrics.json`
records `best_val_log_loss=1.0121814648626144`, `max_length=2048`, and the
ModernBERT base model. Its execution log independently prints `val_log_loss`
and `Best fold 1 validation log loss` as `1.01218`.

### ModernBERT frozen-head public score

The Kaggle CLI competition-submission history reports a public score of
`1.07855` for the submission dated 2026-05-12 with description “ModernBERT head
baseline full run.” The pulled `try-sj-11` notebook is the matching full
ModernBERT frozen-head baseline workflow. The older local `submission_023`
smoke log is separate and reports `1.14524`; it should not be used as the score
evidence for `1.07855`.

## Scope and artifact inventory

The complete inventory is in [`artifact-inventory.csv`](../experiments/artifact-inventory.csv).

### Must remain private or excluded

- `.kaggle/kaggle.json` credential file;
- raw train/test competition data;
- private Kaggle dataset package;
- Gemma adapter weights and tokenizer files;
- offline wheel packages;
- generated submission files and run outputs;
- local virtual environment;
- machine-specific paths and private Kaggle identifiers where not necessary.

### Candidate public evidence

- selected notebooks after safety review and output normalization;
- reusable source logic after path/configuration cleanup;
- configuration files with public model identifiers and no credentials;
- metric summaries with explicit provenance and scope;
- official external links and model/data licensing notes;
- small synthetic fixtures and deterministic tests.

## Repository and source-state findings

- The supplied source repository has no commits and no configured remote.
- The source `.gitignore` excludes the credential file, raw data, virtual environment, and some generated outputs, but it does not comprehensively exclude all large/private Kaggle artifacts.
- Several operational scripts contain absolute machine-specific paths and hard-coded private kernel/dataset references.
- The source contains duplicated implementation across scripts, notebooks, and generated Kaggle packages.
- The target public repository must choose one canonical implementation boundary before migration.

## Phase 1 decision gate

Phase 1 is complete as an audit. Before Phase 2, we need decisions on:

1. Whether the public repository should expose the Kaggle submission notebooks or only sanitized reusable source plus links.
2. Whether to include the older DeBERTa and constant baselines in the main README or keep them in the experiment history.
