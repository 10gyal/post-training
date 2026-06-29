from dataclasses import dataclass, fields
import random
import torch
from torch.utils.data import DataLoader
from tensordict import TensorDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS

from policy_gradients.config import Config
from policy_gradients.utils import apply_reward_kl, compute_log_probs, compute_rewards


@dataclass
class Experience:
    sequence_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    advantages: torch.Tensor | None = None
    rewards: TensorDict | None = None

    def to(self, device: torch.device):
        field_names = [f.name for f in fields(self)]
        moved_tensors = {
            name: getattr(self, name).to(device)
            for name in field_names
            if getattr(self, name) is not None
        }

        return Experience(**moved_tensors)


class RollOutEngine:
    def __init__(
        self,
        cfg: Config,
        dataset: ProceduralDataset,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        ref_model: AutoModelForCausalLM,
        val_model: AutoModelForCausalLM,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.val_model = val_model
        self.cpu_device = torch.device("cpu")

        self.tokenizer.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        self.tokenizer.padding_side = "left"

        self.generation_config = GenerationConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            do_sample=True,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self._data_loader = DataLoader(
            dataset=dataset,
            batch_size=cfg.prompts_per_step,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=lambda x: x,
        )

    def _generate_experience(self, entry: dict) -> Experience:
        entries = [entry for _ in range(self.cfg.num_rollouts)]

        device = self.model.device
        message_templates = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]},
                    {"role": "user", "content": entry["question"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for entry in entries
        ]

        model_inputs = self.tokenizer(
            message_templates,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(device)

        sequence_ids = self.model.generate(
            **model_inputs, generation_config=self.generation_config
        )
        completion_ids = sequence_ids[:, model_inputs["input_ids"].shape[1] :]
        completions = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        self.print_sample = {
            "question": entry["question"],
            "answer": str(entry["answer"]),
            "completion": random.choice(completions),
        }

        action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        action_mask[:, model_inputs["input_ids"].shape[1] :] = True
        action_mask[sequence_ids == self.tokenizer.pad_token_id] = False
        action_mask = action_mask[:, 1:]
        attention_mask = sequence_ids != self.tokenizer.pad_token_id
        lens = action_mask.sum(dim=1).tolist()

        rewards = compute_rewards(
            entries, completions, lens, self.dataset, self.cfg, device
        )

        log_probs_old = compute_log_probs(self.model, sequence_ids, attention_mask)
        log_probs_ref = compute_log_probs(self.ref_model, sequence_ids, attention_mask)

        # kl divergence
        rewards = apply_reward_kl(
            rewards, log_probs_old, log_probs_ref, action_mask, self.cfg
        )

        # no advantages for now
        advantages = rewards

        return Experience(
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            rewards=rewards,
            log_probs_old=log_probs_old,
            log_probs_ref=log_probs_ref,
            advantages=advantages,
        ).to(self.cpu_device)
