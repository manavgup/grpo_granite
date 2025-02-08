# GRPO Granite

This project implements Group Relative Policy Optimization (GRPO) training for IBM's Granite 3.1-2b language model, focusing on mathematical reasoning (GSM8K) and question-answering (RAGBench) capabilities.

## Key Features

### Training Architecture
- Multi-dataset GRPO implementation
- DeepSpeed ZeRO-3 optimization
- VLLM acceleration with dedicated GPU
- Tmux-based session management
- Automatic checkpoint recovery

### Reward Structure
Simplified binary reward approach based on DeepSeek paper:

#### GSM8K Mathematical Reasoning
- Format Reward (0.5):
  - Strict pattern matching for reasoning and answer structure
- Correctness Reward (2.0):
  - Exact match with expected answer
  - No partial credit

#### RAGBench Question-Answering
- Format Reward (0.5):
  - Pattern matching for think/answer structure
- Relevance Reward (2.0):
  - Binary reward based on groundedness and context relevance
  - Thresholds: 0.7 for both metrics

## Technical Requirements
- Python 3.9+
- 8x GPUs (24GB+ each):
  - 7 GPUs for training
  - 1 GPU dedicated to VLLM
- 32GB+ RAM

## Dependencies
```
datasets==3.2.0
transformers==4.48.3
trl==0.14.0
deepspeed
vllm
wandb
math_verify
```

## Setup & Installation
1. Clone repository:
```bash
git clone [repository-url]
cd grpo-granite
```

2. Run setup:
```bash
bash setup.sh
```

3. Configure environment:
```bash
# .env file
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
HF_REPO_ID=your_username/your_model_name
```

## Usage

### Training
```bash
# GSM8K Training (default)
./train_granite.sh

# RAGBench Training
./train_granite.sh ragbench
```

### Training Parameters
- Batch size: 1 per GPU
- Gradient accumulation: 8 steps
- Learning rate: 3e-6
- Max steps: 1000
- Checkpoint frequency: every 50 steps
- VLLM memory utilization: 70%

## Project Structure
```
src/
├── common.py              # Shared utilities
├── base_trainer.py        # Base GRPO implementation
├── train_gsm8k.py        # GSM8K trainer with math rewards
├── train_ragbench.py     # RAGBench trainer with QA rewards
└── zero3.yml             # DeepSpeed config
```

## Monitoring
```bash
# View logs
tail -f logs/train_logs_*.out

# Check tmux session
tmux attach -t granite_training_*

# List checkpoints
ls -l outputs/granite-3.1-2b-*/checkpoint-*
```

## Training Performance
Simplified reward functions provide:
- Faster training iterations
- Clearer success signals
- More stable convergence
- Reduced computational overhead

## License & Attribution
- Licensed under Apache 2.0
- Base model: IBM Granite 3.1-2b
- Reward structure inspired by DeepSeek GRPO paper