import argparse
from pathlib import Path

import torch
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from config import DEFAULT_CONFIG_PATH, Config, load_config
from util import (
    console,
    compute_loss,
    create_dataloader,
    load_model,
    print_chat_templates,
    print_generation_examples,
    print_model_config,
    resolve_device,
    seed_everything,
)


def main(cfg: Config | None = None, config_path: str | Path = DEFAULT_CONFIG_PATH):
    cfg = load_config(config_path) if cfg is None else cfg
    seed_everything(cfg.training.seed)

    device = resolve_device(cfg.hardware.device, cfg.hardware.model_device_id)
    console.print(f"[bold]Device:[/bold] {device}")

    model, tokenizer = load_model(cfg, device)

    print_model_config(cfg)
    print_chat_templates("Chat templates:", tokenizer)
    print_generation_examples("Before training:", model, tokenizer)

    dataloader = create_dataloader(cfg, tokenizer)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = max(1, len(dataloader) * cfg.training.num_epochs)
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.training.num_epochs):
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("loss={task.fields[loss]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1}/{cfg.training.num_epochs}",
                total=len(dataloader),
                loss="n/a",
            )
            for batch in dataloader:
                batch = batch.to(device)
                loss = compute_loss(model, batch)
                loss.backward()

                clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                progress.update(
                    task,
                    advance=1,
                    loss=f"{loss.detach().item():.4f}",
                )

        print_generation_examples(
            f"After epoch {epoch + 1}/{cfg.training.num_epochs}:",
            model,
            tokenizer,
        )
        if epoch + 1 < cfg.training.num_epochs:
            model.train()

    model.eval()

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    main(config_path=args.config)
