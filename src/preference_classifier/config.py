"""Serializable configuration for deterministic and grouped experiments."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    model_name: str
    split_strategy: str = "grouped"
    split_remainder: int = 0
    split_modulus: int = 5
    max_length_train: int = 2048
    max_length_inference: int = 2048
    n_folds: int = 3
    group_key: str = "prompt"
    label_smoothing: float = 0.0
    learning_rate: float = 2e-5
    num_epochs: int = 2
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = ()
    optimizer: str = "adamw_torch"
    swap_augmentation: bool = True
    swap_tta: bool = True
    seed: int = 42
    use_4bit: bool = False

    @property
    def max_length(self) -> int:
        """Backward-compatible alias for the inference token budget."""
        return self.max_length_inference

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        normalized = dict(values)
        if "lora_r" in normalized and "lora_rank" not in normalized:
            normalized["lora_rank"] = normalized.pop("lora_r")
        if "max_length" in normalized:
            normalized.setdefault("max_length_train", normalized["max_length"])
            normalized.setdefault("max_length_inference", normalized.pop("max_length"))
        if "evaluation_rule" in normalized:
            rule = str(normalized.pop("evaluation_rule")).replace(" ", "")
            if rule != "id%5==0":
                raise ValueError("Only the documented id % 5 == 0 rule is supported")
            normalized.setdefault("split_strategy", "id_modulo")
        if "evaluation_remainder" in normalized:
            normalized["split_remainder"] = normalized.pop("evaluation_remainder")
        if "num_folds" in normalized:
            normalized["n_folds"] = normalized.pop("num_folds")
        normalized.pop("fold_index", None)
        if "lora_target_modules" in normalized:
            normalized["lora_target_modules"] = tuple(normalized["lora_target_modules"])
        fields = set(cls.__dataclass_fields__)
        unknown = set(normalized) - fields
        if unknown:
            raise ValueError(f"Unknown experiment config keys: {sorted(unknown)}")
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> ExperimentConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install the project base dependencies to load YAML configs") from exc
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise TypeError("Experiment config must contain a YAML mapping")
    return ExperimentConfig.from_mapping(values)
