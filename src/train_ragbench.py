#!/usr/bin/env python
import re
import json
from typing import List, Dict, Any

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

SYSTEM_PROMPT = """Given a question and some context documents, provide an accurate and well-supported answer.
Use the following format for your response:

<|think|>
Analyze the documents for relevant information.
Identify key facts and evidence.
Determine how to structure the response.
<|think|>
[Your well-structured answer here]
"""

def parse_reasoning_response(text: str) -> dict:
    pattern = r"<\|think\|>\s*(.*?)\s*<\|think\|>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking_content": "", "response": text}
    return {"thinking_content": match.group(1).strip(), "response": match.group(2).strip()}

def get_completion_content(completion: dict) -> str:
    return completion[0]["content"]

def parse_responses(completions: List[dict]) -> List[dict]:
    return [parse_reasoning_response(get_completion_content(c)) for c in completions]

# Reward Functions
def factual_accuracy_reward(prompts, completions, answer, **kwargs) -> List[float]:
    parsed_responses = parse_responses(completions)
    groundedness = float(kwargs.get('trulens_groundedness', 0))
    adherence = float(kwargs.get('gpt3_adherence', 0))
    rewards = [2.0 * (groundedness * 0.7 + adherence * 0.3) for _ in parsed_responses]
    return rewards

def context_relevance_reward(prompts, completions, answer, **kwargs) -> List[float]:
    parsed_responses = parse_responses(completions)
    ragas_relevance = float(kwargs.get('ragas_context_relevance', 0))
    gpt3_relevance = float(kwargs.get('gpt3_context_relevance', 0))
    trulens_relevance = float(kwargs.get('trulens_context_relevance', 0))
    
    rewards = []
    for _ in parsed_responses:
        relevance_score = (
            ragas_relevance * 0.4 +
            gpt3_relevance * 0.4 +
            trulens_relevance * 0.2
        )
        rewards.append(relevance_score)
    return rewards

def source_utilization_reward(prompts, completions, answer, **kwargs) -> List[float]:
    parsed_responses = parse_responses(completions)
    utilization = float(kwargs.get('utilization_score', 0))
    gpt35_util = float(kwargs.get('gpt35_utilization', 0))
    
    rewards = []
    for _ in parsed_responses:
        util_score = (utilization * 0.6 + gpt35_util * 0.4)
        rewards.append(util_score)
    return rewards

def completeness_reward(prompts, completions, answer, **kwargs) -> List[float]:
    parsed_responses = parse_responses(completions)
    completeness = float(kwargs.get('completeness_score', 0))
    rewards = [0.5 * completeness for _ in parsed_responses]
    return rewards

def format_reasoning_reward(prompts, completions, answer, **kwargs) -> List[float]:
    parsed_responses = parse_responses(completions)
    rewards = [0.5 if r["thinking_content"] and r["response"] else 0.0 
              for r in parsed_responses]
    return rewards

# Log rewards and example responses
def log_rewards(prompts, completions, answer, **kwargs):
    rewards = {
        "factual_accuracy": factual_accuracy_reward(prompts, completions, answer, **kwargs),
        "context_relevance": context_relevance_reward(prompts, completions, answer, **kwargs),
        "source_utilization": source_utilization_reward(prompts, completions, answer, **kwargs),
        "completeness": completeness_reward(prompts, completions, answer, **kwargs),
        "format_reasoning": format_reasoning_reward(prompts, completions, answer, **kwargs),
    }
    
    example_response = get_completion_content(completions[0])
    example_parsed = parse_reasoning_response(example_response)
    example_answer = answer[0]
    example_prompt = prompts[0][-1]['content']
    
    print(
        f"-" * 50
        + f"\nExample prompt:\n{example_prompt}\n"
        + f"-" * 10
        + f"\nExample response:\n{example_response}\n"
        + f"-" * 10
        + f"\nExample answer:\n{example_answer}\n"
        + f"-" * 10
        + f"\nRewards:\n{json.dumps(rewards, indent=2)}"
    )

# Initialize reward functions with logging
reward_funcs = [
    format_reasoning_reward,
    factual_accuracy_reward,
    context_relevance_reward,
    source_utilization_reward,
    completeness_reward,
]

def prepare_ragbench_example(example: Dict[str, Any]) -> Dict[str, Any]:
    # Prepare documents string
    documents = example.get('documents', [])
    docs_str = "\n\n".join(documents)
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {example['question']}\n\nContext:\n{docs_str}"},
        ],
        "answer": example['response'],
        "metrics": {
            "trulens_groundedness": example.get('trulens_groundedness', 0),
            "gpt3_adherence": example.get('gpt3_adherence', 0),
            "ragas_context_relevance": example.get('ragas_context_relevance', 0),
            "gpt3_context_relevance": example.get('gpt3_context_relevance', 0),
            "trulens_context_relevance": example.get('trulens_context_relevance', 0),
            "utilization_score": example.get('utilization_score', 0),
            "gpt35_utilization": example.get('gpt35_utilization', 0),
            "completeness_score": example.get('completeness_score', 0),
        }
    }

def load_data(split="train") -> Dataset:
    data = load_dataset("rungalileo/ragbench", "covidqa")[split]
    return data.map(prepare_ragbench_example)

def main(training_args, model_args):
    # Load the RAGBench dataset
    data = load_data()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=data,
    )

    # Train and save
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)