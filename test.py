"""
Interactive test interface for DecisionModel.

Usage:
    python test.py                      # untrained head (random answers)
    python test.py --ckpt decision_model.pt     # load trained weights (state_dict)

Enter a query, then the choices one per line; an empty line runs the model.
"""
import argparse
import string

import torch
from transformers import AutoTokenizer

from backbone import DeviceType
from decision_model import DecisionModel
from prompt_constructor import PromptConstructor

MODEL_NAME = "Qwen/Qwen3.5-4B"
OUT_VECTOR_SIZE = 256
OFFSETS = [16, 33]


def read_choices() -> list[str]:
    choices = []
    while True:
        line = input(f"  {string.ascii_uppercase[len(choices)]}: ").strip()
        if not line:
            return choices
        choices.append(line)


def label_choices(choices: list[str]) -> list[str]:
    # Match the dataset format: "A: ...", "B: ..."
    return [f"{string.ascii_uppercase[i]}: {c}" for i, c in enumerate(choices)]


@torch.no_grad()
def predict(dec: DecisionModel, tokenizer, query: str, choices: list[str]):
    prompt = PromptConstructor.construct_prompt(query, label_choices(choices))
    enc = tokenizer(prompt, return_tensors="pt")
    _, probs, choice = dec.get_choice(enc["input_ids"], len(choices), enc["attention_mask"])
    return probs[0].float().cpu(), choice.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="path to a saved DecisionModel state_dict")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()

    device = DeviceType.GPU if torch.cuda.is_available() and not args.cpu else DeviceType.CPU
    print(f"Loading {MODEL_NAME} on {device.name}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dec = DecisionModel(MODEL_NAME, OUT_VECTOR_SIZE, False, OFFSETS, device)

    if args.ckpt:
        # Load to CPU first; load_state_dict copies into the GPU params in place.
        state = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = dec.load_state_dict(state, strict=False)
        print(f"Loaded {args.ckpt} (missing: {len(missing)}, unexpected: {len(unexpected)})")
    else:
        print("WARNING: no --ckpt given, decision head is untrained -> answers are random.")
    dec.eval()

    print("\nType a query, then choices one per line (empty line = done). Ctrl+C / Ctrl+D to quit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            print("Choices:")
            choices = read_choices()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return

        if len(choices) < 2:
            print("Need at least 2 choices.")
            continue
        if len(choices) > 26:
            print("Too many choices (max 26).")
            continue

        probs, idx = predict(dec, tokenizer, query, choices)
        print(f"\n-> Answer: {string.ascii_uppercase[idx]}: {choices[idx]}")
        for i, (c, p) in enumerate(zip(choices, probs.tolist())):
            bar = "#" * round(p * 30)
            mark = "*" if i == idx else " "
            print(f" {mark} {string.ascii_uppercase[i]}: {c:<20} {p:6.1%} {bar}")


if __name__ == "__main__":
    main()
