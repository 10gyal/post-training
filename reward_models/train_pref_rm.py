import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch.utils.data import DataLoader
from transformers import AutoModel

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

    device = "cuda" if torch.cuda.is_available() else "mps"

    tokenizer = load_tokenizer(base_model)

    ds = prepare_dataset(dataset_name, tokenizer, max_length, limit)
    console.print(f"[bold]Dataset Size:[/bold] {len(ds)}")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(ds) > batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    console.print(f"[bold]Loading model:[/bold] {base_model}")
    model = BTModel(model_name=base_model).to(device)
    console.print(
        f"[bold]Trainable parameters:[/bold] {model.count_trainable_params() / 1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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


main()
