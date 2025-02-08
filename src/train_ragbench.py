#!/usr/bin/env python
"""
RAGBench-specific GRPO training implementation with simplified rewards.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datasets import load_dataset, Dataset

from base_trainer import BaseGRPOTrainer
from common import Completion, RewardFuncKwargs, log_rewards, get_completion_content

class RAGBenchExample(BaseModel):
    """Pydantic model for RAGBench example validation."""
    question: str = Field(..., description="The input question")
    documents: List[str] = Field(..., description="Context documents")
    response: str = Field(..., description="Reference response")
    trulens_groundedness: float = Field(0.0, description="TruLens groundedness score")


class RAGBenchTrainer(BaseGRPOTrainer):
    """Trainer for RAGBench using simplified reward approach."""
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

    def format_reward_func(
        self,
        completions: List[Dict[str, List[Completion]]],
        **kwargs: RewardFuncKwargs
    ) -> List[float]:
        """Reward function that checks if completion follows think-answer format.
        
        Args:
            completions: List of completion dictionaries
            kwargs: Additional reward function arguments
            
        Returns:
            List of reward scores (0.5 for correct format, 0.0 otherwise)
        """
        pattern = r"<\|think\|>\n.*?\n<\|think\|>\n.*"
        responses = [get_completion_content(c) for c in completions]
        return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

    def groundedness_reward_func(
        self,
        prompts: List[Dict],
        completions: List[Dict],
        answer: List[str],
        **kwargs: RewardFuncKwargs
    ) -> List[float]:
        """Reward function based on TruLens groundedness evaluation.
        
        Args:
            prompts: List of prompt dictionaries
            completions: List of completion dictionaries
            answer: List of reference answers
            kwargs: Additional arguments including trulens_groundedness score
            
        Returns:
            List of reward scores (2.0 for high groundedness, 0.0 otherwise)
        """
        groundedness = float(kwargs.get('trulens_groundedness', 0))
        return [2.0 if groundedness > 0.7 else 0.0 for _ in completions]

    def get_reward_functions(self) -> List:
        return [
            self.format_reward_func,        # Format (0.5)
            self.groundedness_reward_func,  # Factual accuracy (2.0)
        ]

    def prepare_ragbench_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare RAGBench example for training.
        
        Args:
            example: Raw example from dataset
            
        Returns:
            Processed example with prompt, answer, and metrics
        """
        # Validate example using Pydantic model
        validated = RAGBenchExample(
            question=example['question'],
            documents=example.get('documents', []),
            response=example['response'],
            trulens_groundedness=example.get('trulens_groundedness', 0)
        )
        docs_str = "\n\n".join(validated.documents)
        
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {validated.question}\n\nContext:\n{docs_str}"},
            ],
            "answer": validated.response,
            "metrics": {
                "trulens_groundedness": validated.trulens_groundedness
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
