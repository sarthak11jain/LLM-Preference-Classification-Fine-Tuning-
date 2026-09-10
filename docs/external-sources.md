# Public sources and attribution register

Public pages are used for task definitions, model documentation, and the
project's recorded Gemma-2 QLoRA run. Result status is stated explicitly so a
reported notebook value is not confused with a fresh rerun.

| Source | Purpose in the project | Phase 1 note |
|---|---|---|
| [Kaggle competition overview](https://www.kaggle.com/competitions/llm-classification-finetuning/overview) | Defines the paired-response preference task, three target probabilities, notebook submission workflow, and log-loss evaluation | Official competition source |
| [Kaggle data page](https://www.kaggle.com/competitions/llm-classification-finetuning/data) | Data access and competition data license context | The page identifies CC BY-NC 4.0; verify current terms before redistribution |
| [ModernBERT-large model card](https://huggingface.co/answerdotai/ModernBERT-large) | Model architecture, long-context capability, model identifier, Apache-2.0 model license, and citation | Official Answer.AI/Hugging Face source |
| [Gemma-2 9B model card](https://huggingface.co/google/gemma-2-9b) | Model identifier, access conditions, Gemma terms, model usage, and citation | Official Google/Hugging Face source; access requires acceptance of terms |
| [Hugging Face PEFT documentation](https://huggingface.co/docs/peft) | LoRA/adapter implementation context | Official library documentation |
| [ModernBERT paper](https://arxiv.org/abs/2412.13663) | Technical background and citation for ModernBERT | External research reference |

## Public Gemma-2 QLoRA run record

The following public notebooks are the source implementation and result record
for the primary individual Gemma-2 workflow. The repository uses their reported
configuration and metrics, while clearly marking them as notebook-reported
because this repository does not rerun the full GPU job:

- [Gemma-2 QLoRA training notebook](https://www.kaggle.com/code/duohanwang/training-gemma-2-9b-4-bit-qlora-fine-tunin-25a55e)
- [Gemma-2 QLoRA inference notebook](https://www.kaggle.com/code/duohanwang/inference-gemma-2-9b-4-bit-qlora-6b251d)

The exact configuration, reported values, downloaded-artifact hashes, and
limitations are captured in
[`evidence/gemma2_qlora_public_notebook_run.json`](../evidence/gemma2_qlora_public_notebook_run.json).

## Publication rules

- Cite the competition and model cards in the public README.
- Keep fresh validation, notebook-reported results, and leaderboard results in
  separate evidence scopes.
- Record the exact model revision and applicable license/terms when a model is used.
- Do not redistribute competition data or model weights without confirming the applicable terms.
