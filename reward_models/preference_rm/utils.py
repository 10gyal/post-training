from pathlib import Path
import yaml

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from reward_models.preference_rm.config import Config


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
        formatted_record.append(f"{r['role']}: {r['content']}\n")
    return "\n".join(formatted_record)


def format_prompt_response(prompt: str, response: str) -> str:
    return f"user: {prompt}\nassistant: {response}"


def prompt_response_to_messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def tokenize_record(
    messages: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
    fallback_text: str | None = None,
):
    """Tokenize with chat template when available, otherwise use local text format."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            max_length=max_length,
            truncation=True,
            add_generation_prompt=False,
            return_dict=True,
        )

    return tokenizer(
        fallback_text if fallback_text is not None else format_record(messages),
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )


def prepare_dataset(
    dataset_name,
    tokenizer: AutoTokenizer,
    max_length,
    limit=10,
    split="train",
) -> Dataset:
    """dictionary of tokenized chosen and rejected records"""
    ds = load_dataset(dataset_name, split=split)

    ds = ds.select(range(limit))

    records = []
    for dp in ds:
        chosen = dp.get("chosen", [])
        rejected = dp.get("rejected", [])
        prompt = dp.get("prompt", "")

        if isinstance(chosen, list):
            chosen_messages = chosen
            rejected_messages = rejected
            chosen_fallback = format_record(chosen_messages)
            rejected_fallback = format_record(rejected_messages)
        elif isinstance(chosen, str):
            chosen_messages = prompt_response_to_messages(prompt, chosen)
            rejected_messages = prompt_response_to_messages(prompt, rejected)
            chosen_fallback = format_prompt_response(prompt, chosen)
            rejected_fallback = format_prompt_response(prompt, rejected)
        else:
            raise TypeError(f"Unsupported preference record type: {type(chosen)}")

        chosen = tokenize_record(
            chosen_messages,
            tokenizer,
            max_length,
            fallback_text=chosen_fallback,
        )

        rejected = tokenize_record(
            rejected_messages,
            tokenizer,
            max_length,
            fallback_text=rejected_fallback,
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


def prepare_reward_bench_dataset(
    tokenizer: AutoTokenizer, subset: str | None = None, max_length: int = 1024
) -> Dataset:
    ds = load_dataset("allenai/reward-bench-2", split="test")
    if subset:
        ds = ds.filter(lambda ex: ex["subset"] == subset)

    records = []
    for row in ds:
        prompt = row["prompt"]
        chosen = row["chosen"][-1]
        rejects = row["rejected"]

        chosen_record = prompt_response_to_messages(prompt, chosen)
        chosen_tokenized = tokenize_record(
            chosen_record,
            tokenizer,
            max_length,
            fallback_text=format_prompt_response(prompt, chosen),
        )

        for r in rejects:
            rejected_record = prompt_response_to_messages(prompt, r)
            reject_tokenized = tokenize_record(
                rejected_record,
                tokenizer,
                max_length=max_length,
                fallback_text=format_prompt_response(prompt, r),
            )

            records.append(
                {
                    "chosen_ids": chosen_tokenized["input_ids"],
                    "chosen_mask": chosen_tokenized["attention_mask"],
                    "rejected_ids": reject_tokenized["input_ids"],
                    "rejected_mask": reject_tokenized["attention_mask"],
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
