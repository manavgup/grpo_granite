#!/usr/bin/env python
"""
GRPO Training Script for GSM8K Mathematical Reasoning

This script implements Group Relative Policy Optimization (GRPO) training
for the Granite-3.1-2b model on the GSM8K dataset, focusing on mathematical
reasoning with explicit step-by-step explanations.

The training process uses multiple reward components:
- Format adherence (XML structure)
- Reasoning quality (length and structure)
- Answer correctness (with math verification)
"""

import re
import json
import torch
from typing import List, Dict, Optional, Union, TypedDict, Tuple
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from math_verify import parse, verify

# Type definitions
class Completion(TypedDict):
    content: str

class RewardFuncKwargs(TypedDict, total=False):
    additional_metrics: Dict[str, float]

@dataclass
class ExtractedContent:
    reasoning: str
    answer: str

SYSTEM_PROMPT: str = """Reason step by step and provide your answer in the following format:

<reasoning>
[Your detailed step-by-step reasoning here]
</reasoning>
<answer>
[Your final numerical answer]
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """
    Extracts answer content from XML-formatted text.
    
    Args:
        text: Input text containing XML tags
        
    Returns:
        Extracted answer string
        
    Note:
        Returns original text if extraction fails
    """
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        return text

def extract_hash_answer(text: str) -> Optional[str]:
    """
    Extracts answer following '####' marker in GSM8K format.
    
    Args:
        text: Input text containing hash-marked answer
        
    Returns:
        Extracted answer string or None if not found
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def count_xml(text: str) -> float:
    """
    Calculates reward score based on XML formatting and reasoning quality.
    
    Args:
        text: Input text to evaluate
        
    Returns:
        Float reward value between 0 and 2.5
        
    Components:
        - Base structure (up to 1.0)
        - Reasoning quality (up to 1.0)
        - Step markers (up to 0.5)
    """
    count: float = 0.0
    
    # Base structure rewards
    if text.count("<reasoning>\n") == 1:
        count += 0.25
    if text.count("\n</reasoning>\n") == 1:
        count += 0.25
    if text.count("\n<answer>\n") == 1:
        count += 0.25
    if text.count("\n</answer>") == 1:
        count += 0.25

    # Penalize extra content
    extra_content = text.split("\n</answer>\n")[-1]
    count -= len(extra_content) * 0.001

    # Quality rewards for reasoning
    try:
        reasoning = text.split("<reasoning>\n")[1].split("\n</reasoning>")[0]
        words = len(reasoning.split())
        # Additional reward for detailed reasoning (up to 1.0)
        count += min(words / 50, 1.0)
        
        # Reward for step-by-step structure
        step_markers = ["step", "first", "then", "next", "finally"]
        if any(marker in reasoning.lower() for marker in step_markers):
            count += 0.5
    except:
        pass

    return count

def format_reward_func(completions: List[Dict[str, List[Completion]]], 
                      **kwargs: RewardFuncKwargs) -> List[float]:
    """
    Calculates format and reasoning quality rewards for completions.
    
    Args:
        completions: List of model outputs
        **kwargs: Additional reward configuration
        
    Returns:
        List of reward values for each completion
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def correctness_reward_func(prompts: List[Dict[str, str]], 
                          completions: List[Dict[str, List[Completion]]], 
                          answer: List[str], 
                          **kwargs: RewardFuncKwargs) -> List[float]:
    """
    Calculates correctness rewards for answers with reasoning.
    
    Args:
        prompts: Input prompts
        completions: Model outputs
        answer: Ground truth answers
        **kwargs: Additional reward configuration
        
    Returns:
        List of reward values for each completion
    """
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Log for monitoring
    print('-'*20, 
          f"Question:\n{q}", 
          f"\nAnswer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0]}", 
          f"\nExtracted:\n{extracted_responses[0]}")
    
    # Only reward correct answers that include reasoning
    rewards: List[float] = []
    for response, extracted in zip(responses, extracted_responses):
        has_reasoning = "<reasoning>" in response and "</reasoning>" in response
        try:
            parsed_answer = parse(extracted)
            parsed_target = parse(answer[0])
            is_correct = verify(parsed_target, parsed_answer)
            reward = 2.0 if is_correct and has_reasoning else 0.0
        except:
            reward = 0.0
        rewards.append(reward)
    
    return rewards

def load_data(split: str = "train") -> Dataset:
    """
    Loads and preprocesses GSM8K dataset.
    
    Args:
        split: Dataset split to load ("train" or "test")
        
    Returns:
        Processed dataset with prompts and answers
    """
    data = load_dataset('openai/gsm8k', 'main')[split]
    return data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']},
        ],
        'answer': extract_hash_answer(x['answer'])
    })

def main() -> None:
    """
    Main training function.
    
    Handles:
    - Argument parsing
    - Dataset loading
    - Model initialization
    - Training configuration
    - Training execution
    """
    # Parse command line arguments
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()

    # Load dataset
    dataset = load_data()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,      # Format and reasoning quality (up to 2.5)
            correctness_reward_func  # Correctness with reasoning (2.0)
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Train and save
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()