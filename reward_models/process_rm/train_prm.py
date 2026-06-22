import argparse
import math
import sys
from pathlib import Path

import random
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from reward_models.base_rm import BaseRM

console = Console()


BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "trl-lib/prm800k"  # clean preprocessed version of openai/prm800k
DEFAULT_SAMPLES = 2000
STEP_SEPARATOR = "\n<step>\n"
PRM_CLASS_VALUES = [False, True]  # Bad, Good
PRM_CLASS_TO_IDX = {value: idx for idx, value in enumerate(PRM_CLASS_VALUES)}
TEST_SAMPLES = 1000

# hyperparameters
DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_TOKENS = 5500
DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 2e-5
DEFAULT_GRAD_ACCUM_STEPS = 8
DEFAULT_EPOCHS = 1
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_SEED = 42
DEFAULT_WANDB_PROJECT = "process_rm"
DEFAULT_WANDB_ONLINE = True
DEFAULT_LOG_EVERY = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


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
    split: str = "train",
    limit: int = DEFAULT_SAMPLES,
    max_steps_per_sample: int = DEFAULT_MAX_STEPS,
    max_tokens_per_sample: int = DEFAULT_MAX_TOKENS,
):
    if isinstance(split, int):
        limit = split
        split = "train"

    stream = load_dataset(dataset_name, split=split, streaming=True)

    sep_token_ids = tokenizer(STEP_SEPARATOR, add_special_tokens=False)["input_ids"]
    len_sep_token_ids = len(sep_token_ids)

    records = []
    for prompt_id, sample in enumerate(stream):
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
            step_label_values = []

            for step_text, lbl in zip(chunk_steps, chunk_labels, strict=True):
                step_payload = step_text.strip() + STEP_SEPARATOR
                encoded = tokenizer(step_payload, add_special_tokens=False)["input_ids"]

                # make sure that the separator is not tokenized differently
                # assert (
                #     encoded[-len_sep_token_ids:] == sep_token_ids
                # ), "separator tokenized differently"

                input_ids.extend(encoded)
                attention_mask.extend([1] * len(encoded))

                # Only label the step terminator token
                step_labels = [-100] * len(encoded)
                cls_id = PRM_CLASS_TO_IDX.get(int(lbl), PRM_CLASS_TO_IDX[0])
                step_labels[-1] = cls_id
                label_ids.extend(step_labels)
                step_label_values.append(PRM_CLASS_VALUES[cls_id])

            # Skip super long traces
            if len(input_ids) > max_tokens_per_sample:
                continue

            records.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids,
                    "prompt_id": prompt_id,
                    "chunk_start": start,
                    "prompt": prompt,
                    "steps": chunk_steps,
                    "step_labels": step_label_values,
                }
            )

    return Dataset.from_list(records[:limit])


def collate_fn(batch, tokenizer: AutoTokenizer, include_metadata: bool = False):
    max_len = max(len(item["input_ids"]) for item in batch)
    inputs = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(inputs)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for idx, item in enumerate(batch):
        length = len(item["input_ids"])
        inputs[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        attn[idx, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long)

    result = {"input_ids": inputs, "attention_mask": attn, "labels": labels}

    if include_metadata:
        result["prompt_ids"] = [
            item.get("prompt_id", idx) for idx, item in enumerate(batch)
        ]
        result["chunk_starts"] = [item.get("chunk_start", 0) for item in batch]
        result["prompts"] = [item.get("prompt", "") for item in batch]
        result["steps"] = [item.get("steps", []) for item in batch]
        result["step_labels"] = [item.get("step_labels", []) for item in batch]

    return result


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
            labels: Per-token class labels (0/1 for steps, -100 for masked)

        Returns:
            loss: Cross-entropy loss on step tokens
            logits: Per-token class logits [batch, seq_len, 2]
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
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_online: bool = DEFAULT_WANDB_ONLINE,
    log_every: int = DEFAULT_LOG_EVERY,
    finish_wandb: bool = True,
) -> tuple[ProcessRewardModel, AutoTokenizer]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_every = max(1, log_every)

    wandb_run = None
    if wandb_online:
        wandb_run = wandb.init(
            project=wandb_project,
            mode="online",
            config={
                "base_model": base_model,
                "dataset_name": dataset_name,
                "samples": samples,
                "batch_size": batch_size,
                "lr": lr,
                "grad_accum_steps": grad_accum_steps,
                "epochs": epochs,
                "warmup_ratio": warmup_ratio,
                "seed": seed,
                "device": str(device),
                "wandb_online": wandb_online,
                "log_every": log_every,
            },
        )

    tokenizer = load_tokenizer(base_model)

    data = prepare_prm_dataset(dataset_name, tokenizer, "train", samples)
    train_generator = make_generator(seed)

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(data) > batch_size,
        generator=train_generator,
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

                    avg_loss = accum_loss / accum_batches
                    acc = accum_correct / max(1, accum_tokens)
                    display_loss = avg_loss
                    display_acc = acc

                    if wandb_run is not None and global_step % log_every == 0:
                        wandb_run.log(
                            {
                                "train/loss": avg_loss,
                                "train/accuracy": acc,
                                "train/step_accuracy": acc,
                                "train/scored_steps": accum_tokens,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                    accum_loss = 0.0
                    accum_correct = 0.0
                    accum_tokens = 0.0
                    accum_batches = 0
                    global_step += 1

                progress.update(
                    task,
                    advance=1,
                    loss=f"{display_loss:.4f}",
                    acc=f"{display_acc:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

    if wandb_run is not None and finish_wandb:
        wandb_run.finish()

    return model, tokenizer


def eval_prm(
    model: ProcessRewardModel,
    tokenizer: AutoTokenizer,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
    limit: int = TEST_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, float]:
    device = next(model.parameters()).device
    eval_ds = prepare_prm_dataset(dataset_name, tokenizer, split, limit)

    if len(eval_ds) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "majority_baseline": float("nan"),
            "prompt_exact_match": float("nan"),
            "steps": 0.0,
            "target_bad": 0.0,
            "target_good": 0.0,
            "pred_bad": 0.0,
            "pred_good": 0.0,
            "accuracy_bad": float("nan"),
            "accuracy_good": float("nan"),
        }

    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, include_metadata=True),
    )

    total_correct = 0
    total_steps = 0
    target_counts = torch.zeros(len(PRM_CLASS_VALUES), dtype=torch.long)
    pred_counts = torch.zeros(len(PRM_CLASS_VALUES), dtype=torch.long)
    correct_counts = torch.zeros(len(PRM_CLASS_VALUES), dtype=torch.long)
    prompt_exact = {}

    model.eval()
    with torch.no_grad():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("acc={task.fields[acc]}"),
            TextColumn("steps={task.fields[steps]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Eval {split}",
                total=len(loader),
                acc="n/a",
                steps="0",
            )

            for batch in loader:
                tensor_batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if torch.is_tensor(value)
                }
                _, logits = model(
                    input_ids=tensor_batch["input_ids"],
                    attention_mask=tensor_batch["attention_mask"],
                )
                mask = tensor_batch["labels"] != -100
                targets = tensor_batch["labels"][mask]
                preds = logits[mask].argmax(dim=-1)

                correct_mask = preds == targets
                total_correct += correct_mask.sum().item()
                total_steps += targets.numel()
                target_counts += torch.bincount(
                    targets.detach().cpu(), minlength=len(PRM_CLASS_VALUES)
                )
                pred_counts += torch.bincount(
                    preds.detach().cpu(), minlength=len(PRM_CLASS_VALUES)
                )
                correct_counts += torch.bincount(
                    targets[correct_mask].detach().cpu(),
                    minlength=len(PRM_CLASS_VALUES),
                )

                row_preds = logits.argmax(dim=-1)
                for row_idx, prompt_id in enumerate(batch["prompt_ids"]):
                    row_mask = mask[row_idx]
                    if not row_mask.any():
                        continue
                    row_correct = (
                        row_preds[row_idx][row_mask]
                        == tensor_batch["labels"][row_idx][row_mask]
                    )
                    chunk_exact = bool(row_correct.all().item())
                    prompt_exact[prompt_id] = (
                        prompt_exact.get(prompt_id, True) and chunk_exact
                    )

                progress.update(
                    task,
                    advance=1,
                    acc=f"{total_correct / max(1, total_steps):.3f}",
                    steps=str(total_steps),
                )

    class_accuracies = torch.full(
        (len(PRM_CLASS_VALUES),), float("nan"), dtype=torch.float
    )
    present_classes = target_counts > 0
    class_accuracies[present_classes] = (
        correct_counts[present_classes].float() / target_counts[present_classes].float()
    )
    balanced_accuracy = class_accuracies[present_classes].mean().item()
    majority_baseline = target_counts.max().item() / max(1, total_steps)
    prompt_exact_match = (
        sum(prompt_exact.values()) / len(prompt_exact) if prompt_exact else float("nan")
    )

    return {
        "accuracy": total_correct / max(1, total_steps),
        "balanced_accuracy": balanced_accuracy,
        "majority_baseline": majority_baseline,
        "prompt_exact_match": prompt_exact_match,
        "steps": float(total_steps),
        "target_bad": float(target_counts[PRM_CLASS_TO_IDX[False]].item()),
        "target_good": float(target_counts[PRM_CLASS_TO_IDX[True]].item()),
        "pred_bad": float(pred_counts[PRM_CLASS_TO_IDX[False]].item()),
        "pred_good": float(pred_counts[PRM_CLASS_TO_IDX[True]].item()),
        "accuracy_bad": class_accuracies[PRM_CLASS_TO_IDX[False]].item(),
        "accuracy_good": class_accuracies[PRM_CLASS_TO_IDX[True]].item(),
    }


def print_eval_metrics(metrics: dict[str, float]) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")

    accuracy = metrics["accuracy"]
    if accuracy == accuracy:
        table.add_row("Step accuracy", f"[bold green]{accuracy:.2%}[/bold green]")
        table.add_row("Balanced accuracy", f"{metrics['balanced_accuracy']:.2%}")
        table.add_row("Majority baseline", f"{metrics['majority_baseline']:.2%}")
        table.add_row("Prompt exact match", f"{metrics['prompt_exact_match']:.2%}")
        table.add_row("Scored steps", f"{metrics['steps']:.0f}")
        table.add_row(
            "Targets bad/good",
            f"{metrics['target_bad']:.0f} / {metrics['target_good']:.0f}",
        )
        table.add_row(
            "Preds bad/good",
            f"{metrics['pred_bad']:.0f} / {metrics['pred_good']:.0f}",
        )
        table.add_row("Bad accuracy", f"{metrics['accuracy_bad']:.2%}")
        table.add_row("Good accuracy", f"{metrics['accuracy_good']:.2%}")
    else:
        table.add_row("Step accuracy", "[bold yellow]n/a[/bold yellow]")
        table.add_row("Balanced accuracy", "nan")
        table.add_row("Majority baseline", "nan")
        table.add_row("Prompt exact match", "nan")
        table.add_row("Scored steps", "0")
        table.add_row("Targets bad/good", "0 / 0")
        table.add_row("Preds bad/good", "0 / 0")
        table.add_row("Bad accuracy", "nan")
        table.add_row("Good accuracy", "nan")

    console.print()
    console.print(Panel(table, title="Process RM Eval", border_style="green"))


def demo_scoring(
    model: ProcessRewardModel,
    tokenizer: AutoTokenizer,
    limit: int = TEST_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    dataset_name: str = DATASET_NAME,
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    seed_everything(seed)

    test_ds = prepare_prm_dataset(dataset_name, tokenizer, "test", limit)

    if len(test_ds) == 0:
        console.print("[bold yellow]No test samples were prepared.[/bold yellow]")
        return []

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=make_generator(seed),
        collate_fn=lambda b: collate_fn(b, tokenizer, include_metadata=True),
    )

    results = []
    sample_counter = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {
                key: value.to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            _, logits = model(**tensor_batch)
            probs = torch.softmax(logits, dim=-1).detach().float().cpu()

            # Identifying positions of step terminator tokens (where label != -100)
            labels = tensor_batch["labels"].detach().cpu()
            batch_rows, _ = labels.shape

            for i in range(batch_rows):
                sample_idx = sample_counter
                sample_counter += 1
                sample_logits = probs[i]  # [seq_len, n_classes]
                sample_labels = labels[i]  # [seq_len]
                step_positions = (sample_labels != -100).nonzero(as_tuple=True)[0]
                steps = batch["steps"][i]
                step_labels = batch["step_labels"][i]

                for step_idx, token_pos in enumerate(step_positions.tolist()):
                    label_id = sample_labels[token_pos].item()
                    target = PRM_CLASS_VALUES[label_id]
                    prob_vec = sample_logits[token_pos].tolist()
                    probs_by_class = {
                        cls: prob_vec[PRM_CLASS_TO_IDX[cls]] for cls in PRM_CLASS_VALUES
                    }
                    pred_id = max(range(len(prob_vec)), key=prob_vec.__getitem__)
                    prediction = PRM_CLASS_VALUES[int(pred_id)]
                    step_text = steps[step_idx] if step_idx < len(steps) else ""
                    raw_target = (
                        step_labels[step_idx] if step_idx < len(step_labels) else target
                    )

                    results.append(
                        {
                            "sample": sample_idx,
                            "step": step_idx + 1,
                            "text": step_text,
                            "target": raw_target,
                            "prediction": prediction,
                            "prob_bad": probs_by_class[False],
                            "prob_good": probs_by_class[True],
                        }
                    )

    table = Table(title="Process RM Step Scores")
    table.add_column("Row", justify="right")
    table.add_column("Sample", justify="right")
    table.add_column("Step", justify="right")
    table.add_column("Target")
    table.add_column("Pred")
    table.add_column("P(bad)", justify="right")
    table.add_column("P(good)", justify="right")
    table.add_column("Text", overflow="fold")

    for row_idx, row in enumerate(results, start=1):
        text = str(row["text"]).strip().replace("\n", " ")
        if len(text) > 140:
            text = text[:137].rstrip() + "..."
        target = "good" if row["target"] else "bad"
        prediction = "good" if row["prediction"] else "bad"
        table.add_row(
            str(row_idx),
            str(row["sample"]),
            str(row["step"]),
            target,
            prediction,
            f"{row['prob_bad']:.3f}",
            f"{row['prob_good']:.3f}",
            escape(text),
        )

    console.print()
    console.print(table)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a process RM.")
    parser.add_argument(
        "--evalonly",
        action="store_true",
        help="Run aggregate step-token eval without training.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-online",
        action="store_true",
        default=DEFAULT_WANDB_ONLINE,
        help="Prompt for W&B login in Python, then log online.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Log every N optimizer steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for model initialization and shuffled loaders.",
    )

    return parser.parse_args()


def log_eval_metrics(metrics: dict[str, float]) -> None:
    wandb.log(
        {
            "eval/accuracy": metrics["accuracy"],
            "eval/step_accuracy": metrics["accuracy"],
            "eval/balanced_accuracy": metrics["balanced_accuracy"],
            "eval/majority_baseline": metrics["majority_baseline"],
            "eval/prompt_exact_match": metrics["prompt_exact_match"],
            "eval/scored_steps": metrics["steps"],
            "eval/target_bad": metrics["target_bad"],
            "eval/target_good": metrics["target_good"],
            "eval/pred_bad": metrics["pred_bad"],
            "eval/pred_good": metrics["pred_good"],
            "eval/accuracy_bad": metrics["accuracy_bad"],
            "eval/accuracy_good": metrics["accuracy_good"],
        }
    )


def main() -> None:
    args = parse_args()

    if args.wandb_online:
        wandb.login()

    if args.evalonly:
        seed_everything(args.seed)
        if args.wandb_online:
            wandb.init(
                project=args.wandb_project,
                mode="online",
                config={
                    "base_model": BASE_MODEL,
                    "dataset_name": DATASET_NAME,
                    "eval_only": True,
                    "seed": args.seed,
                },
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = load_tokenizer(BASE_MODEL)
        model = ProcessRewardModel(model_name=BASE_MODEL).to(device)
        metrics = eval_prm(model, tokenizer)
        if args.wandb_online:
            log_eval_metrics(metrics)
            wandb.finish()
        print_eval_metrics(metrics)
        return

    model, tokenizer = train_prm(
        wandb_project=args.wandb_project,
        wandb_online=args.wandb_online,
        log_every=max(1, args.log_every),
        seed=args.seed,
        finish_wandb=False,
    )

    metrics = eval_prm(model, tokenizer)
    if args.wandb_online:
        log_eval_metrics(metrics)
        wandb.finish()
    print_eval_metrics(metrics)


if __name__ == "__main__":
    main()
