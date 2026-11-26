'''
Takes an prompt and produce logits (or representations)
May convert causal attention to bidirectionals.
'''

from enum import Enum
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BatchEncoding,
    BitsAndBytesConfig
)
from transformers.modeling_outputs import BaseModelOutput

import torch
import torch.nn as nn

class BackboneType(Enum):
    CAUSAL = 1
    EMBED = 2

class DeviceType(Enum):
    CPU = 1
    GPU = 2

class Backbone(nn.Module):
    def __init__(self, model_name: str, model_type: BackboneType, grad: bool, device: DeviceType = DeviceType.CPU):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, device_map="cpu" if device == DeviceType.CPU else "cuda")
        self.model.requires_grad_(grad)
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = text_cfg.hidden_size
        self.model_type = model_type
        self.grad = grad

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        with torch.set_grad_enabled(self.grad and torch.is_grad_enabled()):
            outputs: BaseModelOutput = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, -1, :]