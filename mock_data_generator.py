import json
import random
import string

random.seed(11)
NUM_OF_SAMPLE = 50000
choices_numerical_distance = 10
num_of_choices = 8
DATA = []

for ind in range(NUM_OF_SAMPLE):
    loperand = random.randint(-100, 100)
    roperand = random.randint(-100, 100)
    operation = random.choice(['+', '-', '*'])
    choices = []

    if operation == '+':
        gtruth = loperand + roperand
    elif operation == '-':
        gtruth = loperand - roperand
    elif operation == '*':
        gtruth = loperand * roperand

    gtrplace = random.randint(0, num_of_choices - 1)

    for ch_ind in range(num_of_choices):
        if ch_ind == gtrplace:
            choices.append(gtruth)
        dist = random.randint(-1 * choices_numerical_distance, choices_numerical_distance)
        choices.append(gtruth + dist)

    DATA.append({
        "type": "choice",
        "query": f"What is {loperand} {operation} {roperand} = ?",
        "choices": [
            f"{string.ascii_uppercase[ch_ind]}: {choices[ch_ind]}"
            for ch_ind in range(num_of_choices)
        ],
        "gold_ind": gtrplace
    })

# Create JSONL file
with open('mock.jsonl', 'w') as f:
    for record in DATA:
        f.write(json.dumps(record) + '\n')