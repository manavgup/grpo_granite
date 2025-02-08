"""
Base trainer implementation for GRPO fine-tuning.
"""

import torch
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

from common import Completion, RewardFuncKwargs

class BaseGRPOTrainer(ABC):
    def __init__(self):
        self.parser = TrlParser((GRPOConfig, ModelConfig))
        self.training_args, self.model_args = self.parser.parse_args_and_config()
        
    @abstractmethod
    def load_data(self, split: str = "train") -> Dataset:
        """Load and preprocess dataset."""
        pass
    
    @abstractmethod
    def get_reward_functions(self) -> List:
        """Get list of reward functions."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for the specific task."""
        pass
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """Initialize and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def train(self) -> None:
        """Execute training process."""
        # Load dataset
        dataset = self.load_data()
        
        # Setup tokenizer
        tokenizer = self.setup_tokenizer()
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=self.model_args.model_name_or_path,
            processing_class=tokenizer,
            reward_funcs=self.get_reward_functions(),
            args=self.training_args,
            train_dataset=dataset,
        )
        
        # Train and save
        trainer.train()
        trainer.save_model(self.training_args.output_dir)