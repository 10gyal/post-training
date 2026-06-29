import platform
import os
import random
import re
import numpy as np
from typing import Any

from reasoning_gym.dataset import ProceduralDataset
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensordict import TensorDict
import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.utils import extract_answer

from policy_gradients.config import Config


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(cfg):
    specs = [
        DatasetSpec(name=s.name, weight=s.weight, config=s.config)
        for s in cfg.data.specs
    ]
    return rg.create_dataset(
        "composite", size=cfg.data.size, seed=cfg.seed, datasets=specs
    )


def get_attn_implementation() -> str:
    """Determine the best attention implementation for this platform.

    Returns 'flash_attention_2' on x86_64 with flash-attn installed,
    otherwise 'sdpa' (PyTorch's native SDPA, faster on DGX Spark/Blackwell).
    """
    if platform.machine() != "x86_64":
        return "sdpa"  # aarch64 / DGX Spark - use SDPA with cuDNN

    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model(model_name: str, device_map: Any, gradient_checkpointing: bool = True):
    """Load model and tokenizer with automatic attention implementation selection."""
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    # Many decoder-only models (LLaMA, GPT-2) don't define pad_token
    # Set it to eos_token to enable batch padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, tokenizer


def get_ref_model(model_name: str, device_map: Any, beta: float):
    """Load reference model for KL penalty (only if beta > 0)."""
    if not beta:
        return None
    ref_model, _ = load_model(model_name, device_map, gradient_checkpointing=False)
    ref_model.eval()
    return ref_model


def get_val_model(
    model_name: str, device_map: Any, loss: str, gradient_checkpointing: bool = True
):
    """Load value model."""
    if loss not in ["ppo"]:
        return None
    val_model, _ = load_model(model_name, device_map, gradient_checkpointing)
    val_model.lm_head = nn.Linear(
        val_model.lm_head.in_features,
        1,
        bias=False,
        device=val_model.device,
        dtype=torch.bfloat16,
    )
    return val_model


def _correctness_reward(
    dataset: ProceduralDataset, completions: list[str], entries: list[dict]
):
    scores = [
        float(dataset.score_answer(extract_answer(completion), entry))
        for completion, entry in zip(completions, entries, strict=True)
    ]

    return scores


def _format_reward(completions: list[str]) -> list[float]:
    def count_tags(text: str) -> float:
        count = 0.0
        if re.search(r"\s*<think>\s", text):
            count += 0.25
        if re.search(r"\s*</think>\s", text):
            count += 0.25
        if re.search(r"\s*<answer>\s", text):
            count += 0.25
        if re.search(r"\s*</think>\s", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


def compute_rewards(
    entries: list[dict],
    completions: list[str],
    lens: list[int],
    dataset: ProceduralDataset,
    cfg: Config,
    device: torch.device,
) -> TensorDict:
    correctness = _correctness_reward(dataset, completions, entries)
    # Skip penalties for REINFORCE

    format_adherence = _format_reward(completions)

    total = [
        c + cfg.format_weight * f
        for c, f in zip(correctness, format_adherence, strict=True)
    ]

    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(
        -1
    )

    rewards = TensorDict(
        {
            "total": to_tensor(total),
            "correctness": to_tensor(correctness),
            "format_adherence": to_tensor(format_adherence),
        },
        batch_size=[len(entries)],
    )

    rewards["binary"] = (
        rewards["correctness"] * rewards["format_adherence"] == 1.0
    ).float()

    return rewards


def compute_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor | None:
    if not model:
        return None

    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(
        model.device
    )
    output = model(
        input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False
    )
    values = output.logits[:, :-1, :].squeeze(-1).to(torch.float32)
    return values


def apply_reward_kl(
    rewards,
    log_probs_old,
    log_probs_ref,
    action_mask,
    cfg,
):
    return rewards
