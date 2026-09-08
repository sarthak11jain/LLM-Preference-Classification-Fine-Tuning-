# Phase 1 — External sources and attribution register

External pages are supporting context and licensing references. They are not
evidence that the project's own metrics were achieved.

| Source | Purpose in the project | Phase 1 note |
|---|---|---|
| [Kaggle competition overview](https://www.kaggle.com/competitions/llm-classification-finetuning/overview) | Defines the paired-response preference task, three target probabilities, notebook submission workflow, and log-loss evaluation | Official competition source |
| [Kaggle data page](https://www.kaggle.com/competitions/llm-classification-finetuning/data) | Data access and competition data license context | The page identifies CC BY-NC 4.0; verify current terms before redistribution |
| [ModernBERT-large model card](https://huggingface.co/answerdotai/ModernBERT-large) | Model architecture, long-context capability, model identifier, Apache-2.0 model license, and citation | Official Answer.AI/Hugging Face source |
| [Gemma-2 9B model card](https://huggingface.co/google/gemma-2-9b) | Model identifier, access conditions, Gemma terms, model usage, and citation | Official Google/Hugging Face source; access requires acceptance of terms |
| [Hugging Face PEFT documentation](https://huggingface.co/docs/peft) | LoRA/adapter implementation context | Official library documentation |
| [ModernBERT paper](https://arxiv.org/abs/2412.13663) | Technical background and citation for ModernBERT | External research reference |

## References reviewed in the original README

The supplied README also lists public Kaggle notebooks used as methodological
references. Their scores and artifacts are external and must not be presented as
this project's results:

- [duohanwang — Gemma-2 QLoRA training](https://www.kaggle.com/code/duohanwang/training-gemma-2-9b-4-bit-qlora-fine-tunin-25a55e)
- [duohanwang — Gemma-2 inference](https://www.kaggle.com/code/duohanwang/inference-gemma-2-9b-4-bit-qlora-6b251d)
- [rivaldofauzan — Gemma LoRA training](https://www.kaggle.com/code/rivaldofauzan/2-4-training-gemma-lora-8-5)
- [rvldfr — LoRA ensemble inference](https://www.kaggle.com/code/rvldfr/3-1-inference-ensemble-lora-4-5)
- [ainidr25 — DoRA ensemble inference](https://www.kaggle.com/code/ainidr25/3-15-inference-ensemble-dora-8-2)
- [David Pupăză — competition notebook](https://www.kaggle.com/code/davidpupaza/llm-classification-finetuning)
- [Carla Cotas — competition notebook](https://www.kaggle.com/code/carlacotas/llm-classification-finetuning-version-3-0)

## Attribution rules for Phase 2

- Cite the competition and model cards in the public README.
- Keep project results separate from external benchmark/reference results.
- Record the exact model revision and applicable license/terms when a model is used.
- Do not redistribute competition data or model weights without confirming the applicable terms.
