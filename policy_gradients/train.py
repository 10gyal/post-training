import torch
from torch.utils.data import DataLoader

from config import Config, load_config
from utils import (
    create_dataset,
    seed_everything,
    load_model,
    get_ref_model,
    get_val_model,
)


def reinforce_loss():
    pass


def main(cfg: Config):
    seed_everything(cfg.seed)

    cpu_device = torch.device("cpu")
    if torch.cuda.is_available():
        model_device = torch.device(f"cuda:{cfg.model_device_id}")
        ref_model_device = torch.device(f"cuda:{cfg.ref_model_device_id}")
        val_model_device = torch.device(f"cuda:{cfg.val_model_device_id}")
    else:
        model_device = ref_model_device = val_model_device = cpu_device

    data = create_dataset(cfg)

    model, tokenizer = load_model(cfg.model_name, model_device)

    ref_model = get_ref_model(cfg.model_name, ref_model_device, cfg.beta)
    val_model = get_val_model(cfg.model_name, val_model_device, cfg.loss)


if __name__ == "__main__":
    cfg = load_config("reinforce.yaml")
    data = create_dataset(cfg)
    for d in data:
        print(d)
        break
