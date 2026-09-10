"""Optional Hugging Face/PEFT model construction.

The heavy dependencies are imported lazily so schema and augmentation utilities
remain usable on CPU-only machines.
"""


def _base_sequence_classifier(model_name: str, num_labels: int, use_4bit: bool):
    try:
        from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra to build LoRA models") from exc

    model_kwargs = {"num_labels": num_labels}
    if use_4bit:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch for 4-bit model loading") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    return AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)


def build_lora_sequence_classifier(model_name: str, *, num_labels: int = 3, **lora_kwargs):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra to build LoRA models") from exc

    model = _base_sequence_classifier(model_name, num_labels, lora_kwargs.get("use_4bit", False))
    target_modules = lora_kwargs.get("target_modules")
    if not target_modules and "modernbert" in model_name.lower():
        target_modules = modernbert_lora_defaults()["target_modules"]
    if not target_modules and "gemma" in model_name.lower():
        target_modules = gemma2_lora_defaults()["target_modules"]
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_kwargs.get("rank", 16),
        lora_alpha=lora_kwargs.get("alpha", 32),
        lora_dropout=lora_kwargs.get("dropout", 0.05),
        target_modules=target_modules,
        modules_to_save=["classifier", "score"],
    )
    return get_peft_model(model, config)


def load_lora_sequence_classifier(checkpoint: str, model_name: str, *, num_labels: int = 3, **kwargs):
    """Load a saved adapter on the configured base model."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra to load LoRA models") from exc
    base = _base_sequence_classifier(model_name, num_labels, kwargs.get("use_4bit", False))
    return PeftModel.from_pretrained(base, checkpoint)


def modernbert_lora_defaults() -> dict:
    return {"target_modules": ["Wqkv", "Wi", "Wo"], "r": 16, "lora_alpha": 32, "lora_dropout": 0.1}


def gemma2_lora_defaults() -> dict:
    return {
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
