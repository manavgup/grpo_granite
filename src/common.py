"""
Common utilities and base classes for GRPO training scripts.
"""

from typing import List, Dict, Optional, Union, TypedDict
from dataclasses import dataclass
import json

# Type definitions
class Completion(TypedDict):
    content: str

class RewardFuncKwargs(TypedDict, total=False):
    additional_metrics: Dict[str, float]

@dataclass
class ExtractedContent:
    reasoning: str
    answer: str

def log_rewards(rewards: Dict[str, List[float]], example_data: Dict) -> None:
    """
    Logs rewards and example data in a consistent format.
    
    Args:
        rewards: Dictionary of reward names and values
        example_data: Dictionary containing example information
    """
    print(
        f"-" * 50
        + f"\nExample prompt:\n{example_data['prompt']}\n"
        + f"-" * 10
        + f"\nExample response:\n{example_data['response']}\n"
        + f"-" * 10
        + f"\nExample answer:\n{example_data['answer']}\n"
        + f"-" * 10
        + f"\nRewards:\n{json.dumps(rewards, indent=2)}"
    )

def get_completion_content(completion: Dict) -> str:
    """Extracts content from completion dictionary."""
    return completion[0]["content"]

def base_reward_func(value: float, scale: float = 1.0) -> float:
    """Base reward calculation with scaling."""
    return value * scale