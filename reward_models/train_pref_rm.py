from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader, dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from utils import load_config, load_tokenizer, prepare_dataset, collate_fn

console = Console()


class BaseRM(nn.Module):
    def __init__(self, model_name, head_dim: int = 1) -> None:
        super().__init__()

        device_map = {"": 0} if torch.cuda.is_available() else None
        self.model = AutoModel.from_pretrained(
            model_name, dtype="bfloat16", device_map=device_map, trust_remote_code=True
        )
        self.model.use_cache = False

        self.head = self._build_head(self.model.config.hidden_size, head_dim)
        self.head = self.head.to(torch.bfloat16)

    def _build_head(self, hidden_size, output_dim):
        return nn.Linear(hidden_size, output_dim, bias=output_dim > 1)

    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BTModel(BaseRM):
    def __init__(self, model_name, head_dim: int = 1) -> None:
        super().__init__(model_name, head_dim)

    def get_reward(self, ids, masks):
        # get the hidden state of the last non-padding token from the sequence
        hidden = self.get_hidden_states(ids, masks)

        seq_lengths = masks.sum(dim=1) - 1  # zero-index lengths
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, seq_lengths]

        reward = self.head(last_hidden).squeeze(-1)
        return reward

    def forward(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        r_chosen = self.get_reward(chosen_ids, chosen_mask)
        r_rejected = self.get_reward(rejected_ids, rejected_mask)

        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        return loss, r_chosen, r_rejected


def eval(model: BTModel, dloader: DataLoader) -> dict[str, float]:
    """Measure preference accuracy and reward statistics on a test split."""
    device = next(model.parameters()).device

    correct, total = 0, 0
    chosen_sum, rejected_sum, margin_sum = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in dloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, r_chosen, r_rejected = model(**batch)
            chosen = r_chosen.detach().float()
            rejected = r_rejected.detach().float()
            margin = chosen - rejected

            correct += (margin > 0).sum().item()
            batch_size = chosen.size(0)
            total += batch_size
            chosen_sum += chosen.sum().item()
            rejected_sum += rejected.sum().item()
            margin_sum += margin.sum().item()

    if total == 0:
        return {
            "accuracy": float("nan"),
            "chosen_reward": float("nan"),
            "rejected_reward": float("nan"),
            "reward_margin": float("nan"),
        }

    return {
        "accuracy": correct / total,
        "chosen_reward": chosen_sum / total,
        "rejected_reward": rejected_sum / total,
        "reward_margin": margin_sum / total,
    }


def print_eval_metrics(metrics: dict[str, float]) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")

    accuracy = metrics["accuracy"]
    if accuracy == accuracy:
        table.add_row("Preference accuracy", f"[bold green]{accuracy:.2%}[/bold green]")
        table.add_row("Mean chosen reward", f"{metrics['chosen_reward']:.4f}")
        table.add_row("Mean rejected reward", f"{metrics['rejected_reward']:.4f}")
        table.add_row("Mean margin", f"{metrics['reward_margin']:+.4f}")
    else:
        table.add_row("Preference accuracy", "[bold yellow]n/a[/bold yellow]")
        table.add_row("Mean chosen reward", "nan")
        table.add_row("Mean rejected reward", "nan")
        table.add_row("Mean margin", "nan")

    console.print()
    console.print(Panel(table, title="Evaluation", border_style="green"))


def main():

    cfg = load_config("config.yaml")
    wandb = None
    if cfg.use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config=cfg.model_dump(),
        )

    base_model = cfg.base_model
    dataset_name = cfg.dataset_name
    limit = cfg.limit
    max_length = cfg.max_length
    batch_size = cfg.batch_size
    lr = cfg.lr
    epochs = cfg.epochs
    warmup_ratio = cfg.warmup_ratio

    device = "cuda" if torch.cuda.is_available() else "mps"

    tokenizer = load_tokenizer(base_model)

    train_ds, test_ds = prepare_dataset(
        dataset_name, tokenizer, max_length, limit, split="train"
    )
    console.print(f"[bold]Dataset Size:[/bold] {len(train_ds)}")
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_ds) > batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    console.print(f"[bold]Loading model:[/bold] {base_model}")
    model = BTModel(model_name=base_model).to(device)
    console.print(
        f"[bold]Trainable parameters:[/bold] {model.count_trainable_params() / 1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = max(1, len(loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(epochs):
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("loss={task.fields[loss]}"),
            TextColumn("chosen={task.fields[chosen]}"),
            TextColumn("rejected={task.fields[rejected]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1}/{epochs}",
                total=len(loader),
                loss="n/a",
                chosen="n/a",
                rejected="n/a",
            )

            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                loss, r_chosen, r_rejected = model(**batch)

                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress.update(
                    task,
                    advance=1,
                    loss=f"{loss.detach().item():.4f}",
                    chosen=f"{r_chosen.detach().float().mean().item():.4f}",
                    rejected=f"{r_rejected.detach().float().mean().item():.4f}",
                )

                if wandb is not None and global_step % cfg.log_every == 0:
                    reward_margin = r_chosen - r_rejected
                    wandb.log(
                        {
                            "train/loss": loss.detach().item(),
                            "train/reward_chosen": r_chosen.detach()
                            .float()
                            .mean()
                            .item(),
                            "train/reward_rejected": r_rejected.detach()
                            .float()
                            .mean()
                            .item(),
                            "train/reward_margin": reward_margin.detach()
                            .float()
                            .mean()
                            .item(),
                            "train/preference_accuracy": (reward_margin > 0)
                            .detach()
                            .float()
                            .mean()
                            .item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                global_step += 1

    if wandb is not None:
        wandb.finish()

    dloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(test_ds) > batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    metrics = eval(model, dloader)

    print_eval_metrics(metrics)


main()
