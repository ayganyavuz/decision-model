import json
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from prompt_constructor import PromptConstructor


class DecisionDataset(Dataset):
    def __init__(self, file: str = "mock.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        prompt = PromptConstructor.construct_prompt(item["query"], item["choices"])
        return {
            "prompt": prompt,
            "gold_ind": item["gold_ind"],
            "type": item["type"],
            "n_choices": len(item["choices"]),
        }


class DecisionCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int | None = None):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [x["prompt"] for x in batch]
        encoded = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([x["gold_ind"] for x in batch], dtype=torch.long),  # (b,)
            "types": [x["type"] for x in batch],
            "n_choices": [x["n_choices"] for x in batch],
        }