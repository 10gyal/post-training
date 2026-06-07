import json
import os
import random
from dataclasses import dataclass
from urllib.request import urlretrieve

import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_CONFIG_PATH, Config, load_config
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100
GENERATION_EXAMPLES = [
    [{"role": "user", "content": "Hi, how was your day?"}],
    [{"role": "user", "content": "What is the sum of 2+2?"}],
]


console = Console()


def resolve_device(device: str, model_device_id: int = 0) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{model_device_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(cfg, device: torch.device | str | None = None):
    device = (
        resolve_device(cfg.hardware.device, cfg.hardware.model_device_id)
        if device is None
        else device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_lm, trust_remote_code=False, use_fast=True
    )

    if tokenizer.chat_template is None:
        if cfg.model.chat_template_tokenizer:
            donor = AutoTokenizer.from_pretrained(
                cfg.model.chat_template_tokenizer,
                trust_remote_code=False,
                use_fast=True,
            )
            tokenizer.chat_template = donor.chat_template

            # Set all the special tokens from donor to tokenizer
            special_token_names = [
                "pad_token",
                "bos_token",
                "eos_token",
                "unk_token",
                "sep_token",
                "cls_token",
                "mask_token",
                "additional_special_tokens",
            ]
            for token_name in special_token_names:
                donor_token = getattr(donor, token_name, None)
                if donor_token is not None:
                    setattr(tokenizer, token_name, donor_token)

        else:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if loop.first and messages[0]['role'] != 'system' %}"
                "{{ 'system\nBelow is an instruction that describes a task. "
                "Write a response that appropriately completes the request."
                "\n<|endoftext|>\n' }}"
                "{% endif %}"
                "{{ message['role'] + '\n' + message['content'] "
                "+ '<|endoftext|>' + '\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ 'assistant\n' }}{% endif %}"
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.hardware.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_lm,
        trust_remote_code=False,
        dtype=dtype,
    ).to(device)

    return model, tokenizer


def generate_response(messages, model, tokenizer, max_new_tokens: int = 512) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )


def generate_example_responses(model, tokenizer, max_new_tokens: int = 512):
    return [
        {
            "messages": messages,
            "response": generate_response(
                messages,
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
            ),
        }
        for messages in GENERATION_EXAMPLES
    ]


def run_generation_examples(config_path=DEFAULT_CONFIG_PATH) -> None:
    cfg = load_config(config_path)
    device = resolve_device(cfg.hardware.device, cfg.hardware.model_device_id)
    model, tokenizer = load_model(cfg, device)
    model.eval()

    for example in generate_example_responses(model, tokenizer):
        prompt = example["messages"][0]["content"]
        print(f"User: {prompt}")
        print(f"Assistant: {example['response']}")
        print()


def print_generation_examples(title: str, model, tokenizer) -> None:
    console.rule(f"[bold]{title}")
    model.eval()

    for i, example in enumerate(generate_example_responses(model, tokenizer), start=1):
        prompt = example["messages"][0]["content"]
        response = example["response"]
        if len(response) > 200:
            response = f"{response[:200]}..."

        console.print(Text(f"Example {i}", style="bold cyan"))
        console.print(Text.assemble(("Prompt: ", "bold green"), prompt))
        console.print(Text.assemble(("Response: ", "bold"), response))
        console.print()

    console.print()


def print_chat_templates(title: str, tokenizer) -> None:
    console.rule(f"[bold]{title}")

    for messages in GENERATION_EXAMPLES:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = messages[0]["content"]
        console.print(
            Panel(
                Text(rendered),
                title=prompt,
                title_align="left",
                border_style="cyan",
            )
        )

    console.print()


def print_model_config(cfg: Config) -> None:
    chat_template_from = cfg.model.chat_template_tokenizer or "Custom"
    if cfg.data.use_local_instruction_data:
        ift_dataset = f"local:{cfg.data.instruction_data_path}"
    elif cfg.data.dataset_subset:
        ift_dataset = (
            f"{cfg.data.ift_dataset}/{cfg.data.dataset_subset}"
            f" ({cfg.data.dataset_split})"
        )
    else:
        ift_dataset = f"{cfg.data.ift_dataset} ({cfg.data.dataset_split})"

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Pretrained LM", cfg.model.pretrained_lm)
    table.add_row("Chat template from", chat_template_from)
    table.add_row("IFT dataset", ift_dataset)
    console.print(Panel(table, title="Run Configuration", border_style="green"))
    console.print()


def encode_example(example, tokenizer: AutoTokenizer, max_length):
    messages = example["messages"]

    # Prompt and full templates are computed separately because chat templates
    # can add role markers and generation prompts contextually.
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
    )

    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )

    if len(full_ids) > max_length:
        return None

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


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        urlretrieve(url, file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def transform_to_messages_wrapped(data):
    examples = []
    for example in data:
        conversation = []
        user_content = ""
        instr = example.get("instruction", "")
        inp = example.get("input", "")
        if instr and inp:
            user_content = instr + "\n" + inp
        elif instr:
            user_content = instr
        if user_content:
            conversation.append({"role": "user", "content": user_content})

        if example.get("output"):
            conversation.append({"role": "assistant", "content": example["output"]})
        if conversation:
            examples.append({"messages": conversation})
    return examples


def load_sft_data(cfg: Config):
    if cfg.data.use_local_instruction_data:
        data = download_and_load_file(
            cfg.data.instruction_data_path,
            cfg.data.instruction_data_url,
        )
        return transform_to_messages_wrapped(data)

    return load_dataset(
        cfg.data.ift_dataset,
        cfg.data.dataset_subset,
        split=cfg.data.dataset_split,
    )


def create_dataloader(cfg: Config, tokenizer: AutoTokenizer):
    raw_dataset = load_sft_data(cfg)
    ds = SFTDataset(raw_dataset, tokenizer, max_length=cfg.data.max_length)

    pad_token_id = tokenizer.pad_token_id

    return DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
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
    shift_logits = output.logits[:, :-1, :].contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


if __name__ == "__main__":
    run_generation_examples()
