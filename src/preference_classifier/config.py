"""Serializable experiment configuration for public training workflows."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ExperimentConfig:
    model_name: str
    max_length: int = 2048
    n_folds: int = 3
    group_key: str = "prompt"
    label_smoothing: float = 0.0
    learning_rate: float = 2e-5
    num_epochs: int = 2
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    swap_augmentation: bool = True
    swap_tta: bool = True
    seed: int = 42
    use_4bit: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        normalized = dict(values)
        if "lora_r" in normalized and "lora_rank" not in normalized:
            normalized["lora_rank"] = normalized.pop("lora_r")
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
        raise ValueError("Experiment config must contain a YAML mapping")
    return ExperimentConfig.from_mapping(values)
