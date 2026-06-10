from pydantic import BaseModel


class Config(BaseModel):
    base_model: str
    dataset_name: str
    limit: int
    max_length: int
    batch_size: int
    lr: float
    epochs: int
    warmup_ratio: float
    use_wandb: bool = False
    wandb_project: str = "learn-rlhf"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    log_every: int = 1
