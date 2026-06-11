import torch
import torch.nn as nn
from transformers import AutoModel


class BaseRM(nn.Module):
    def __init__(self, model_name: str, head_dim: int = 1) -> None:
        super().__init__()

        device_map = {"": 0} if torch.cuda.is_available() else None
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype="bfloat16",
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.use_cache = False

        self.head = self._build_head(self.model.config.hidden_size, head_dim)
        self.head = self.head.to(torch.bfloat16)

    def _build_head(self, hidden_size: int, output_dim: int) -> nn.Module:
        return nn.Linear(hidden_size, output_dim, bias=output_dim > 1)

    def get_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
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
