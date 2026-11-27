from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import DecisionDataset, DecisionCollator
from decision_model import DecisionModel
from backbone import DeviceType
import torch
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 32
LR = 8e-4
SAVE_PATH = "decision_model.pt"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
mock = DecisionDataset("mock.jsonl")
train_dataloader = DataLoader(dataset=mock, batch_size=BATCH_SIZE, collate_fn=DecisionCollator(tokenizer))



it = iter(train_dataloader)
b1 = next(it)

dec = DecisionModel("Qwen/Qwen3.5-4B", 256, True, [16, 33], DeviceType.GPU)
optimizer = optim.AdamW(
    [p for p in dec.parameters() if p.requires_grad],
    lr=LR,
)

loss_F = nn.CrossEntropyLoss()
cnt = 0
dec.train()
try:
    for batch in train_dataloader:
        logits, probs, choice = dec.get_choice(batch["input_ids"], 8, batch["attention_mask"])
        cnt += 1
        # print(f"Choice: {choice}")
        # print(f"Logits: {logits}")
        # print(f"Probs: {probs}")
        # print(f"Gold Probs: {batch["labels"]}")
        labels = batch["labels"].to(logits.device)
        loss = loss_F(logits.float(), labels)

        print(f"cnt: {32 * cnt}/50000 \nloss val: {loss}")
        print("\n\n")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
except KeyboardInterrupt:
    print("Training interrupted, saving current weights...")

# Full model (backbone + decision head). ~9 GB in bf16.
torch.save(dec.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")
