#!/usr/bin/env python
"""
RAGBench-specific GRPO training implementation with simplified rewards.
"""

import re
from typing import List, Dict, Any
from datasets import load_dataset, Dataset

from base_trainer import BaseGRPOTrainer
from common import Completion, RewardFuncKwargs, log_rewards, get_completion_content

class RAGBenchTrainer(BaseGRPOTrainer):
    SYSTEM_PROMPT = """Given a question and some context documents, provide an accurate and well-supported answer.
Use the following format for your response:

<|think|>
Analyze the documents for relevant information.
Identify key facts and evidence.
Determine how to structure the response.
<|think|>
[Your well-structured answer here]
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def format_reward_func(self, completions: List[Dict[str, List[Completion]]], 
                         **kwargs: RewardFuncKwargs) -> List[float]:
        """Simple format check for think-answer structure."""
        pattern = r"<\|think\|>\n.*?\n<\|think\|>\n.*"
        responses = [get_completion_content(c) for c in completions]
        return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

    def groundedness_reward_func(self, prompts: List[Dict], 
                               completions: List[Dict], 
                               answer: List[str], 
                               **kwargs: RewardFuncKwargs) -> List[float]:
        """Binary reward based on trulens groundedness score."""
        groundedness = float(kwargs.get('trulens_groundedness', 0))
        return [2.0 if groundedness > 0.7 else 0.0 for _ in completions]

    def get_reward_functions(self) -> List:
        return [
            self.format_reward_func,        # Format (0.5)
            self.groundedness_reward_func,  # Factual accuracy (2.0)
        ]

    def prepare_ragbench_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare RAGBench example with minimal required metrics."""
        documents = example.get('documents', [])
        docs_str = "\n\n".join(documents)
        
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {example['question']}\n\nContext:\n{docs_str}"},
            ],
            "answer": example['response'],
            "metrics": {
                "trulens_groundedness": example.get('trulens_groundedness', 0)
            }
        }

    def load_data(self, split: str = "train") -> Dataset:
        data = load_dataset("rungalileo/ragbench", "covidqa")[split]
        return data.map(self.prepare_ragbench_example)

def main():
    trainer = RAGBenchTrainer()
    trainer.train()

if __name__ == "__main__":
    main()