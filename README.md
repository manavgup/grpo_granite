# GRPO Granite

This project implements a Generative Reward-Penalized Optimization (GRPO) approach to fine-tune IBM's Granite 3.1-2b language model on the GSM8K dataset for enhanced mathematical reasoning capabilities.

## Key Features

- GRPO implementation with multiple reward functions:
  - Answer accuracy
  - Number formatting
  - Reasoning structure
- Integration with IBM Granite 3.1-2b model
- DeepSpeed ZeRO-3 optimization for efficient training
- VLLM acceleration for faster model inference
- Custom reward functions for evaluating model outputs

## Technical Stack

- Python 3.9+
- Dependencies:
  - datasets==3.2.0
  - transformers==4.48.3
  - trl==0.14.0
  - deepspeed
  - vllm
- Hardware requirements:
  - GPU with at least 24GB of memory
  - 32GB+ RAM

## Installation & Setup

1. Create a Python virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have the necessary configuration files:
   - `src/zero3.yml`: DeepSpeed ZeRO-3 configuration
   - `setup.sh`: Environment setup script

## Usage

To start the training process, run the following command:

```
bash src/train_granite.sh
```

This script will launch the training using the Accelerate library with the provided configuration. The training logs will be saved in the `train_logs_*.out` files.

## Project Structure

- `src/`
  - `train_granite.sh`: Training script
  - `train_gsm8k.py`: Main training implementation
  - `zero3.yml`: DeepSpeed ZeRO-3 configuration
- `requirements.txt`: Python dependencies
- `setup.sh`: Environment setup script

## License & Attribution

This project is licensed under the [Apache 2.0 License](LICENSE).

The Granite 3.1-2b model used in this project is developed and provided by IBM.
