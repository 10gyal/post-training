from dataclasses import dataclass


@dataclass
class Config:
    pretrained_lm: str = "HuggingFaceTB/SmolLM2-135M"
    ift_dataset: str = "HuggingFaceH4/no_robots"
    # the pretrained lm does not come with a chat template
    chat_template_tokenizer: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_split: str = "train"
    max_length: int = 2048

    # training hyperparams
    lr: float = 5.0e-6
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    model_device_id: int = 0

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None
