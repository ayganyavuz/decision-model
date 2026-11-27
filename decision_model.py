import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import Backbone, BackboneType, DeviceType
from typing import List

class DecisionModel(nn.Module):
    def __init__(self, backbone_name: str, out_vector_size: int, grad: bool, offsets: List[int], device: DeviceType = DeviceType.CPU):
        super().__init__()
        self.grad = grad
        self.backbone = Backbone(backbone_name, BackboneType.CAUSAL, self.grad, device)
        self.decision_head = nn.Linear(
            self.backbone.hidden_size,
            out_vector_size,
            bias=False,
            dtype=torch.float32,
            device=self.backbone.model.device,
        )
        self.offsets = offsets # [15, 48]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        encoded = self.backbone.encode(input_ids, attention_mask) # (b, 2560)
        return self.decision_head(encoded.float()) # (b, out_vector_size)

    def get_binary(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        raw_vector = self.forward(input_ids, attention_mask) # (b, out_vector_size)
        binary_logits = raw_vector[:, 0:self.offsets[0]]
        res = F.sigmoid(binary_logits)
        return res

    def get_score(self, input_ids: torch.Tensor, n_scores: int, attention_mask: torch.Tensor | None = None):
        raw_vector = self.forward(input_ids, attention_mask)
        score_logits = raw_vector[:, self.offsets[0]:self.offsets[0] + n_scores]
        values, indices = torch.sort(score_logits, dim=-1)
        values_softmax = F.softmax(values, dim=-1)
        scores = indices.to(values_softmax.dtype) 
        expected_value = (values_softmax * scores).sum(dim=-1)
        return score_logits, expected_value, values_softmax, indices

    def get_choice(self, input_ids: torch.Tensor, n_choices: int, attention_mask: torch.Tensor | None = None):
        raw_vector = self.forward(input_ids, attention_mask)
        choices_logits = raw_vector[:, self.offsets[1]:self.offsets[1] + n_choices]

        values_softmax = F.softmax(choices_logits, dim=-1)
        max_index = torch.argmax(values_softmax, dim=-1)
        return choices_logits, values_softmax, max_index