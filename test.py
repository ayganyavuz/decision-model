from backbone import Backbone, BackboneType
from decision_model import DecisionModel
from prompt_constructor import PromptConstructor
import json

'''
dec = DecisionModel(256, False, [1, 33])

print(dec.get_binary("Bu bina sağlam mı?").item()) 
'''
with open("mock.jsonl", 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]


print("safjsaf")