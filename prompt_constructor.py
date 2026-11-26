from typing import List

class PromptConstructor():
    @staticmethod
    def construct_prompt(query: str, choices: List[str]):
        return f"""You are a decision model, you need to correctly choose from possible choices.
Query: {query}
Choices: {choices}
<decision>
"""
