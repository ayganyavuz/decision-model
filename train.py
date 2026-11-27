from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import DecisionDataset, DecisionCollator
from decision_model import DecisionModel
from backbone import DeviceType
import torch
import torch.nn as nn
import torch.optim as optim
import time

BATCH_SIZE = 16
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
LOG_EVERY = 50
SAVE_PATH = "decision_model.pt"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
mock = DecisionDataset("mock.jsonl")
train_dataloader = DataLoader(dataset=mock, batch_size=BATCH_SIZE, shuffle=True, collate_fn=DecisionCollator(tokenizer))

dec = DecisionModel("Qwen/Qwen3.5-4B", 256, True, [16, 33], DeviceType.GPU)

# Separate LRs: small for the pretrained backbone, large for the freshly initialized head.
optimizer = optim.AdamW([
    {"params": [p for p in dec.backbone.parameters() if p.requires_grad], "lr": BACKBONE_LR},
    {"params": dec.decision_head.parameters(), "lr": HEAD_LR},
])
# Linear warmup to the target LRs over WARMUP_STEPS, then constant.
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / WARMUP_STEPS))

loss_F = nn.CrossEntropyLoss()
total_steps = len(train_dataloader)
step = 0
run_loss, run_correct, run_seen = 0.0, 0, 0

dec.train()
step_time = time.time()
try:
    for batch in train_dataloader:
        n_choices = max(batch["n_choices"])
        logits, probs, choice = dec.get_choice(batch["input_ids"], n_choices, batch["attention_mask"])
        labels = batch["labels"].to(logits.device)
        loss = loss_F(logits.float(), labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        step += 1

        run_loss += loss.item() * labels.size(0)
        run_correct += (choice == labels).sum().item()
        run_seen += labels.size(0)
         
        if step % LOG_EVERY == 0 or step == total_steps:
            time_consumed = time.time() - step_time()
            step_time = time.time()

            print(
                f"step {step}/{total_steps} | samples {step * BATCH_SIZE}/{len(mock)} | "
                f"loss {run_loss / run_seen:.4f} | acc {run_correct / run_seen:.2%} "
                f"(chance {1 / n_choices:.1%})"
                f"{LOG_EVERY * BATCH_SIZE / time_consumed} examples/secon"
            )
            run_loss, run_correct, run_seen = 0.0, 0, 0
except KeyboardInterrupt:
    print("Training interrupted, saving current weights...")

# Full model (backbone + decision head). ~9 GB in bf16.
torch.save(dec.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")
