"""Package-level orchestration for baselines and optional GPU fine-tuning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .augmentation import augment_with_swaps, swap_rows
from .baselines import comparison_texts, fit_tfidf_logistic_regression
from .config import ExperimentConfig
from .data import labels_from_frame, read_csv
from .evaluation import multiclass_log_loss
from .inference import combine_swap_tta
from .models import build_lora_sequence_classifier, load_lora_sequence_classifier
from .splits import configured_splits
from .submission import write_submission


def run_tfidf_validation(train_path: str | Path, config: ExperimentConfig) -> dict:
    """Run the lightweight CPU baseline through the same split dispatcher."""
    frame = read_csv(train_path, training=True)
    split = next(configured_splits(frame, config))
    train_idx, validation_idx = split
    _, _, probabilities = fit_tfidf_logistic_regression(frame.iloc[train_idx], frame.iloc[validation_idx])
    labels = labels_from_frame(frame.iloc[validation_idx])
    return {
        "model": "TF-IDF logistic regression",
        "split_rule": config.split_strategy,
        "rows": len(validation_idx),
        "score": multiclass_log_loss(labels, probabilities),
        "metric": "log_loss",
    }


def make_demo_submission(test_path: str | Path, output_path: str | Path) -> None:
    """Create the CPU demo submission using the public uniform baseline."""
    frame = pd.read_csv(test_path)
    probabilities = np.full((len(frame), 3), 1 / 3, dtype=float)
    write_submission(frame["id"], probabilities, output_path)


def _transformer_components(config: ExperimentConfig):
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra for transformer workflows") from exc
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_lora_sequence_classifier(
        config.model_name,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules) or None,
        use_4bit=config.use_4bit,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return torch, tokenizer, model


def _build_optimizer(model, config: ExperimentConfig, torch):
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("adamw_8bit requires bitsandbytes; use adamw_torch for a fallback") from exc
        return bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "adamw_torch":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


class _PreferenceDataset:
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.encodings = tokenizer(
            comparison_texts(frame),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels_from_frame(frame)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


def _predict_loader(model, loader, device, torch) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def run_transformer_training(
    train_path: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict:
    """Train one or more configured LoRA folds and save metrics/checkpoints."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra for transformer workflows") from exc
    from .training import cross_entropy_loss, seed_everything

    seed_everything(config.seed)
    frame = read_csv(train_path, training=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_records = []
    for fold_number, (train_idx, validation_idx) in enumerate(configured_splits(frame, config), start=1):
        torch, tokenizer, model = _transformer_components(config)
        train_frame = frame.iloc[train_idx]
        if config.swap_augmentation:
            train_frame = augment_with_swaps(train_frame)
        validation_frame = frame.iloc[validation_idx]
        train_data = _PreferenceDataset(train_frame, tokenizer, config.max_length_train)
        validation_data = _PreferenceDataset(validation_frame, tokenizer, config.max_length_inference)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        validation_loader = DataLoader(validation_data, batch_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not config.use_4bit:
            model.to(device)
        elif next(model.parameters()).device.type != "cuda" and torch.cuda.is_available():
            device = next(model.parameters()).device
        optimizer = _build_optimizer(model, config, torch)
        loss_function = cross_entropy_loss(config.label_smoothing)
        best_loss = float("inf")
        for _ in range(config.num_epochs):
            model.train()
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()
                logits = model(**{key: value for key, value in batch.items() if key != "labels"}).logits
                loss = loss_function(logits, batch["labels"])
                loss.backward()
                optimizer.step()
            probs = _predict_loader(model, validation_loader, device, torch)
            if config.swap_tta:
                swapped = _PreferenceDataset(swap_rows(validation_frame), tokenizer, config.max_length_inference)
                swapped_probs = _predict_loader(model, DataLoader(swapped, batch_size=1), device, torch)
                probs = combine_swap_tta(probs, swapped_probs)
            loss = multiclass_log_loss(labels_from_frame(validation_frame), probs)
            if loss < best_loss:
                best_loss = loss
        fold_dir = output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)
        record = {
            "fold": fold_number,
            "model": config.model_name,
            "split_rule": config.split_strategy,
            "validation_rows": len(validation_idx),
            "best_val_log_loss": best_loss,
            "seed": config.seed,
            "train_length": config.max_length_train,
            "inference_length": config.max_length_inference,
            "swap_augmentation": config.swap_augmentation,
            "swap_tta": config.swap_tta,
        }
        (fold_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        fold_records.append(record)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {"config": config.to_dict(), "folds": fold_records}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=list), encoding="utf-8")
    return result


def run_transformer_prediction(
    test_path: str | Path,
    config: ExperimentConfig,
    checkpoint: str | Path,
    output_path: str | Path,
) -> None:
    """Load one saved adapter and write competition-format probabilities."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the [training] extra for transformer workflows") from exc
    test = pd.read_csv(test_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_lora_sequence_classifier(
        str(checkpoint),
        config.model_name,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules) or None,
        use_4bit=config.use_4bit,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config.use_4bit:
        model.to(device)
    elif next(model.parameters()).device.type != "cuda" and torch.cuda.is_available():
        device = next(model.parameters()).device
    labeled_test = test.assign(winner_model_a=1.0, winner_model_b=0.0, winner_tie=0.0)
    dataset = _PreferenceDataset(labeled_test, tokenizer, config.max_length_inference)
    probabilities = _predict_loader(model, DataLoader(dataset, batch_size=1), device, torch)
    write_submission(test["id"], probabilities, output_path)


def require_gpu_pipeline() -> None:
    """Provide one clear failure mode until optional training dependencies load."""
    raise RuntimeError(
        "Gemma-2 and ModernBERT training require the [training] extra, model weights, "
        "and a compatible GPU runtime. Use the Kaggle notebooks or install the extra."
    )
