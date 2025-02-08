#!/usr/bin/env python
"""
GSM8K-specific GRPO training implementation with simplified reward functions.
"""

import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datasets import load_dataset, Dataset

from base_trainer import BaseGRPOTrainer
from common import Completion, RewardFuncKwargs, get_completion_content


class Answer(BaseModel):
    """Pydantic model for answer validation and extraction."""
    value: str = Field(..., description="The numerical answer value")
    reasoning: str = Field(..., description="Step-by-step reasoning")


class GSM8KTrainer(BaseGRPOTrainer):
    """Trainer for GSM8K math problems using simplified reward approach."""
    
    SYSTEM_PROMPT = """Reason step by step and provide your answer in the following format:

<reasoning>
[Your detailed step-by-step reasoning here]
</reasoning>
<answer>
[Your final numerical answer]
</answer>
"""

    def get_system_prompt(self) -> str:
        """Return the system prompt for GSM8K training."""
        return self.SYSTEM_PROMPT

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from response text using simple string operations."""
        try:
            answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
            return answer
        except IndexError:
            return None

    def extract_reference_answer(self, text: str) -> Optional[str]:
        """Extract reference answer from GSM8K dataset format."""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def correctness_reward_func(
        self,
        prompts: List[Dict[str, str]],
        completions: List[Dict[str, List[Completion]]],
        answer: List[str],
        **kwargs: RewardFuncKwargs
    ) -> List[float]:
        """Simple correctness reward based on exact answer match."""
        responses = [get_completion_content(c) for c in completions]
        extracted = [self.extract_answer(r) for r in responses]
        return [1.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

    def int_reward_func(
        self,
        completions: List[Dict[str, List[Completion]]],
        **kwargs: RewardFuncKwargs
    ) -> List[float]:
        """Reward function that checks if the extracted answer is a valid integer."""
        responses = [get_completion_content(c) for c in completions]
        extracted = [self.extract_answer(r) for r in responses]
        return [0.5 if r and r.strip().isdigit() else 0.0 for r in extracted]

    def strict_format_reward_func(
        self,
        completions: List[Dict[str, List[Completion]]],
        **kwargs: RewardFuncKwargs
    ) -> List[float]:
        """Reward function that checks if the completion follows strict XML format."""
        pattern = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>"
        responses = [get_completion_content(c) for c in completions]
        return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

    def soft_format_reward_func(
        self,
        completions: List[Dict[str, List[Completion]]],
        **kwargs: RewardFuncKwargs
    ) -> List[float]:
        """Reward function that checks if the completion follows a less strict format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [get_completion_content(c) for c in completions]
        return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

    def get_reward_functions(self) -> List:
        """Return all reward functions for comprehensive evaluation."""
        return [
            self.correctness_reward_func,    # Answer correctness (1.0)
            self.int_reward_func,            # Integer check (0.5)
            self.strict_format_reward_func,   # Strict format (0.5)
            self.soft_format_reward_func      # Soft format (0.5)
        ]

    def load_data(self, split: str = "train") -> Dataset:
        """Load and preprocess GSM8K dataset."""
        data = load_dataset('openai/gsm8k', 'main')[split]
        return data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']},
            ],
            'answer': self.extract_reference_answer(x['answer'])
        })


def main():
    """Main entry point for GSM8K training."""
    trainer = GSM8KTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
