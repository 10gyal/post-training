from dataclasses import dataclass

from datasets import load_dataset
import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader


IGNORE_INDEX = -100
MAX_LENGTH = 2048


def encode_example(example, tokenizer: PreTrainedTokenizer, max_length):
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


class CFG:
    def __init__(self, dataset_name, dataset_split="train", batch_size=4):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.batch_size = batch_size


def create_dataloader(cfg: CFG, tokenizer: PreTrainedTokenizer):

    raw_dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

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


if __name__ == "__main__":
    test = CFG(
        dataset_name="HuggingFaceH4/no_robots",
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    dataloader = create_dataloader(test, tokenizer)
    for batch in dataloader:
        print(batch)
        break
