from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class ModelConfig:
    pretrained_lm: str
    chat_template_tokenizer: str | None


@dataclass
class DataConfig:
    ift_dataset: str
    dataset_subset: str | None
    dataset_split: str
    max_length: int
    use_local_instruction_data: bool
    instruction_data_path: str
    instruction_data_url: str


@dataclass
class TrainingConfig:
    lr: float
    num_epochs: int
    batch_size: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    seed: int


@dataclass
class HardwareConfig:
    device: str
    bf16: bool
    model_device_id: int


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    hardware: HardwareConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        hardware=HardwareConfig(**raw["hardware"]),
    )
