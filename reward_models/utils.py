from pathlib import Path
import yaml

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from config import Config


def load_config(path: Path) -> "Config":
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with proper padding setup."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_record(record):
    formatted_record = []
    for r in record:
        formatted_record.append(f"{r["role"]}: {r["content"]}\n")
    return "\n".join(formatted_record)


def prepare_dataset(
    dataset_name, tokenizer: AutoTokenizer, max_length, limit=10
) -> Dataset:
    """dictionary of tokenized chosen and rejected records"""
    ds = load_dataset(dataset_name, split="train")

    ds = ds.select(range(limit))

    records = []
    for dp in ds:
        chosen = dp.get("chosen", [])
        rejected = dp.get("rejected", [])
        prompt = dp.get("prompt", "")

        if isinstance(chosen, list):
            chosen_record = format_record(chosen)
            rejected_record = format_record(rejected)
        elif isinstance(chosen, str):
            chosen_record = f"user: {prompt}\nassistant: {chosen}"
            rejected_record = f"user: {prompt}\nassistant: {rejected}"

        chosen = tokenizer(
            chosen_record,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )

        rejected = tokenizer(
            rejected_record,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )

        records.append(
            {
                "chosen_ids": chosen["input_ids"],
                "chosen_mask": chosen["attention_mask"],
                "rejected_ids": rejected["input_ids"],
                "rejected_mask": rejected["attention_mask"],
            }
        )
    return Dataset.from_list(records)


def collate_fn(batch, tokenizer: AutoTokenizer):

    def pad_sequences(
        sequences: list[list[int]],
        pad_value: int,
        return_tensors: bool = True,
    ) -> torch.Tensor | list[list[int]]:
        """Pad sequences to the same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [pad_value] * (max_len - len(seq)))

        if return_tensors:
            return torch.tensor(padded, dtype=torch.long)
        return padded

    return {
        "chosen_ids": pad_sequences(
            [x["chosen_ids"] for x in batch], tokenizer.pad_token_id
        ),
        "chosen_mask": pad_sequences([x["chosen_mask"] for x in batch], 0),
        "rejected_ids": pad_sequences(
            [x["rejected_ids"] for x in batch], tokenizer.pad_token_id
        ),
        "rejected_mask": pad_sequences([x["rejected_mask"] for x in batch], 0),
    }
