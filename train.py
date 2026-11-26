from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import DecisionDataset, DecisionCollator
from decision_model import DecisionModel
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 32
LR = 1e-5

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
mock = DecisionDataset("mock.jsonl")
train_dataloader = DataLoader(dataset=mock, batch_size=2, collate_fn=DecisionCollator(tokenizer))


dec = DecisionModel("Qwen/Qwen3.5-4B", 256, True, [16, 33])

it = iter(train_dataloader)
b1 = next(it)
loss_F = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    [p for p in dec.parameters() if p.requires_grad],
    lr=LR,
)

dec.train()
for batch in train_dataloader:
    logits, probs, choice = dec.get_choice(batch["input_ids"], 8, batch["attention_mask"])
    # print(f"Choice: {choice}")
    # print(f"Logits: {logits}")
    # print(f"Probs: {probs}")
    # print(f"Gold Probs: {batch["labels"]}")
    labels = batch["labels"].to(logits.device)
    loss = loss_F(logits.float(), labels)

    print(f"loss val: {loss}") 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    