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
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    use_wandb: bool = False
    wandb_project: str = "learn-rlhf"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    log_every: int = 1
