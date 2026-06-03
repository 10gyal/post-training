import os
import platform

import numpy as np
from dataclasses import dataclass
from datasets import load_dataset
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from rich.console import Console

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from config import Config

console = Console()


IGNORE_INDEX = -100
MAX_LENGTH = 2048


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_attn_implementation() -> str:
    if platform.machine() != "x86_64":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model(cfg: Config, device: torch.device):
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_lm, trust_remote_code=False
    )

    if tokenizer.chat_template is None and cfg.chat_template_tokenizer:
        donor = AutoTokenizer.from_pretrained(
            cfg.chat_template_tokenizer, trust_remote_code=False
        )
        if donor.chat_template is None:
            raise ValueError(
                f"chat_template_source {cfg.chat_template_tokenizer} has no chat_template."
            )
        tokenizer.chat_template = donor.chat_template

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_lm,
        trust_remote_code=False,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    ).to(device)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, tokenizer


def encode_example(example, tokenizer: AutoTokenizer, max_length):
    messages = example["messages"]

    # apply_chat_template(messages) != apply_chat_template(messages[-1:]) + apply_chat_template(messages[:-1])
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
    )

    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )

    labels = [IGNORE_INDEX] * len(prompt_ids)
    labels += full_ids[len(prompt_ids) :]

    input_ids = full_ids[:max_length]
    labels = labels[:max_length]

    if all(x == IGNORE_INDEX for x in labels):
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@dataclass
class SFTBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> "SFTBatch":
        return SFTBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


class SFTDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_length):
        self.examples = []

        for row in raw_dataset:
            encoded = encode_example(row, tokenizer, max_length)
            if encoded is not None:
                self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collator(examples, pad_token_id):
    max_len = max(ex["input_ids"].size(0) for ex in examples)

    input_ids = []
    attention_mask = []
    labels = []

    for ex in examples:
        ids = ex["input_ids"]
        lbl = ex["labels"]
        pad_len = max_len - ids.size(0)

        input_ids.append(
            torch.cat(
                [
                    ids,
                    torch.full(
                        size=(pad_len,), fill_value=pad_token_id, dtype=torch.long
                    ),
                ]
            )
        )

        attention_mask.append(
            torch.cat(
                [
                    torch.ones(size=(ids.size(0),), dtype=torch.long),
                    torch.zeros(size=(pad_len,), dtype=torch.long),
                ]
            )
        )

        labels.append(
            torch.cat(
                [
                    lbl,
                    torch.full(
                        size=(pad_len,), fill_value=IGNORE_INDEX, dtype=torch.long
                    ),
                ]
            )
        )

    return SFTBatch(
        input_ids=torch.stack(input_ids),
        attention_mask=torch.stack(attention_mask),
        labels=torch.stack(labels),
    )


def create_dataloader(cfg: Config, tokenizer: AutoTokenizer):

    raw_dataset = load_dataset(cfg.ift_dataset, split=cfg.dataset_split)

    ds = SFTDataset(raw_dataset, tokenizer, max_length=MAX_LENGTH)

    pad_token_id = tokenizer.pad_token_id

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collator(batch, pad_token_id),
        num_workers=0,
        pin_memory=False,
    )


def compute_loss(model, batch):
    output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    # shift for next token prediction
    shift_logits = output.logits[:, :-1, :].contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("*"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_epoch_header(epoch_idx: int, total_epochs: int) -> None:
    console.rule(
        f"[bold cyan]Epoch {epoch_idx + 1}/{total_epochs}[/bold cyan]", style="cyan"
    )


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps + 1))
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(Config.chat_template_tokenizer)
    dataloader = create_dataloader(Config, tokenizer)
    for batch in dataloader:
        # test
        print(batch)
        break
