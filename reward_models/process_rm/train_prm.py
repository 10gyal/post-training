import math
import sys
from pathlib import Path

import random
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from reward_models.base_rm import BaseRM

console = Console()


BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "trl-lib/prm800k"  # clean preprocessed version of openai/prm800k
DEFAULT_SAMPLES = 2
STEP_SEPARATOR = "\n<step>\n"
PRM_CLASS_VALUES = [-1, 0, 1]  # Bad, Neutral, Good
PRM_CLASS_TO_IDX = {value: idx for idx, value in enumerate(PRM_CLASS_VALUES)}


# hyperparameters
DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_TOKENS = 5500
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 2e-5
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_EPOCHS = 1
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_SEED = 42


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with proper padding setup."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# prepare the dataset
def prepare_prm_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    limit: int = 10,
    max_steps_per_sample: int = DEFAULT_MAX_STEPS,
    max_tokens_per_sample: int = DEFAULT_MAX_TOKENS,
):
    stream = load_dataset(dataset_name, split="train", streaming=True)

    records = []
    for sample in stream:
        if len(records) == limit:
            break
        prompt = sample.get("prompt", "")
        steps = sample.get("completions", [])
        labels = sample.get("labels", [])

        # simple chat template
        prompt = f"Problem: {prompt}\nReasoning trace:\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Chunk very long traces to avoid OOM
        for start in range(0, len(steps), max_steps_per_sample):
            if len(records) == limit:
                break

            chunk_steps = steps[start : start + max_steps_per_sample]
            chunk_labels = labels[start : start + max_steps_per_sample]

            if not chunk_steps or not chunk_labels:
                continue

            input_ids = list(prompt_ids)
            attention_mask = [1] * len(input_ids)
            label_ids = [-100] * len(input_ids)

            for step_text, lbl in zip(chunk_steps, chunk_labels, strict=True):
                step_payload = step_text.strip() + STEP_SEPARATOR
                encoded = tokenizer(step_payload, add_special_tokens=False)["input_ids"]
                input_ids.extend(encoded)
                attention_mask.extend([1] * len(encoded))

                # Only label the step terminator token
                step_labels = [-100] * len(encoded)
                cls_id = PRM_CLASS_TO_IDX.get(int(lbl), PRM_CLASS_TO_IDX[0])
                step_labels[-1] = cls_id
                label_ids.extend(step_labels)

            # Skip super long traces
            if len(input_ids) > max_tokens_per_sample:
                continue

            records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids,
                }
            )

    return Dataset.from_list(records[:limit])


def collate_fn(batch, tokenizer: AutoTokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    inputs = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(inputs)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for idx, item in enumerate(batch):
        length = len(item["input_ids"])
        inputs[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        attn[idx, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long)

    return {"input_ids": inputs, "attention_mask": attn, "labels": labels}


class ProcessRewardModel(BaseRM):
    """Architecure:
    - Base Model in BF16
    - Linear head mapping hidden states to 3-class logits
    The model ouptuts per-token logits which are trained with cross-entropy loss on step terminator tokens only (all other tokens masked)
    """

    def __init__(self, model_name: str = BASE_MODEL, **kwargs) -> None:
        super().__init__(model_name, head_dim=len(PRM_CLASS_VALUES), **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """Forward pass computing step logits and optional loss.
        Args:
            input_ids: token ids [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Per-token class labels (0/1/2 for steps, -100 for masked)

        Returns:
            loss: Cross-entropy loss on step tokens
            logits: Per-token class logits [batch, seq_len, 3]
        """

        hidden = self.get_hidden_states(input_ids, attention_mask)
        logits = self.head(hidden)

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                loss = F.cross_entropy(logits[mask], labels[mask])
            else:
                loss = logits.sum() * 0

        return loss, logits


# setup the training loop


def train_prm(
    base_model: str = BASE_MODEL,
    dataset_name: str = DATASET_NAME,
    samples: int = DEFAULT_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
    epochs: int = DEFAULT_EPOCHS,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    seed: int = DEFAULT_SEED,
):
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(base_model)

    data = prepare_prm_dataset(dataset_name, tokenizer, samples)

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(data) > batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    model = ProcessRewardModel(model_name=base_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = max(1, math.ceil(len(loader) / grad_accum_steps) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    num_train_batches = len(loader)
    final_accum_steps = num_train_batches % grad_accum_steps or grad_accum_steps
    optimizer.zero_grad()

    model.train()

    for epoch in range(epochs):
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("loss={task.fields[loss]}"),
            TextColumn("acc={task.fields[acc]}"),
            TextColumn("lr={task.fields[lr]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1}/{epochs}",
                total=len(loader),
                loss="n/a",
                acc="n/a",
                lr="n/a",
            )
            accum_loss = 0.0
            accum_correct = 0.0
            accum_tokens = 0.0
            accum_batches = 0

            for batch_idx, batch in enumerate(loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}

                loss, logits = model(**batch)
                loss_item = loss.detach().float().item()

                accum_divisor = (
                    final_accum_steps
                    if batch_idx > num_train_batches - final_accum_steps
                    else grad_accum_steps
                )

                (loss / accum_divisor).backward()

                accum_loss += loss_item
                mask = batch["labels"] != -100
                preds = logits[mask].argmax(dim=-1)
                correct = (preds == batch["labels"][mask]).sum().item()
                tokens = mask.sum().item()
                accum_correct += correct
                accum_tokens += tokens
                accum_batches += 1
                display_loss = accum_loss / accum_batches
                display_acc = accum_correct / max(1, accum_tokens)

                should_step = (
                    batch_idx % grad_accum_steps == 0 or batch_idx == num_train_batches
                )

                if should_step:
                    optimizer.step()
                    scheduler.step()

                    optimizer.zero_grad()
                    global_step += 1

                    avg_loss = accum_loss / accum_batches
                    acc = accum_correct / max(1, accum_tokens)
                    display_loss = avg_loss
                    display_acc = acc

                    accum_loss = 0.0
                    accum_correct = 0.0
                    accum_tokens = 0.0
                    accum_batches = 0

                progress.update(
                    task,
                    advance=1,
                    loss=f"{display_loss:.4f}",
                    acc=f"{display_acc:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )


# todo: evals

if __name__ == "__main__":
    train_prm()
