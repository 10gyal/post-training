from typing import Any

import yaml
from pydantic import BaseModel


class DatasetSpec(BaseModel):
    """Specification for a single dataset in the training mixture."""

    name: str
    weight: int = 1
    config: dict[str, Any] = {}


class DataConfig(BaseModel):
    """Configuration for the training data."""

    specs: list[DatasetSpec]
    size: int = 3000


class Config(BaseModel):
    data: DataConfig
    seed: int = 42

    # Reward shaping
    format_weight: float = 0.5


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
