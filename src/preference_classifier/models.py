"""Optional Hugging Face/PEFT model construction.

The heavy dependencies are imported lazily so schema and augmentation utilities
remain usable on CPU-only machines.
"""


def build_lora_sequence_classifier(model_name: str, *, num_labels: int = 3, **lora_kwargs):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra to build LoRA models") from exc

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_kwargs.get("rank", 16),
        lora_alpha=lora_kwargs.get("alpha", 32),
        lora_dropout=lora_kwargs.get("dropout", 0.05),
        target_modules=lora_kwargs.get("target_modules"),
        modules_to_save=["classifier", "score"],
    )
    return get_peft_model(model, config)


def modernbert_lora_defaults() -> dict:
    return {"target_modules": ["Wqkv", "Wi", "Wo"], "r": 16, "lora_alpha": 32, "lora_dropout": 0.1}


def gemma2_lora_defaults() -> dict:
    return {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "r": 16, "lora_alpha": 32, "lora_dropout": 0.1}
