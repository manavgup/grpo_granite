# GRPO Granite

This project implements a Group Relative Policy Optimization (GRPO) approach to fine-tune IBM's Granite 3.1-2b language model on the GSM8K dataset for enhanced mathematical reasoning capabilities.

## Key Features
- GRPO implementation with enhanced reward functions:
  - Strong reasoning emphasis (up to 2.5 reward)
  - Step-by-step explanation rewards
  - Answer accuracy validation
  - Format structure verification
- Integration with IBM Granite 3.1-2b model
- DeepSpeed ZeRO-3 optimization for efficient training
- VLLM acceleration for faster model inference
- Robust training with automatic recovery:
  - Checkpoint-based resumption
  - Training state monitoring
  - Automatic retry on failure
- Tmux session management for reliable long-running training
- Automatic model upload to Hugging Face Hub
- Comprehensive logging and monitoring
- Weights & Biases integration for experiment tracking

## Technical Stack
- Python 3.9+
- Dependencies:
  - datasets==3.2.0
  - transformers==4.48.3
  - trl==0.14.0
  - deepspeed
  - vllm
  - wandb
  - math_verify
  - tmux
- Hardware requirements:
  - 8x GPUs with at least 24GB of memory each (7 for training, 1 for VLLM)
  - 32GB+ RAM

## Installation & Setup
1. Clone the repository and navigate to the project directory:
```bash
git clone [repository-url]
cd grpo-granite
```

2. Run the setup script to create and configure the Python environment:
```bash
bash setup.sh
```

3. Create a `.env` file in the project root with your API keys:
```bash
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
HF_REPO_ID=your_username/your_model_name
```

## Usage
To start the training process:
```bash
bash src/train_granite3.1-2b.sh
```

This script will:
1. Create a tmux session for reliable training
2. Load environment variables from `.env`
3. Initialize Weights & Biases logging
4. Train the model using DeepSpeed ZeRO-3 optimization
5. Automatically recover from failures (up to 3 retries)
6. Upload the trained model to Hugging Face Hub
7. Maintain comprehensive logs

## Project Structure
- `src/`
  - `train_granite3.1-2b.sh`: Enhanced training script with tmux and recovery
  - `train_gsm8k.py`: Main training implementation with typed rewards
  - `zero3.yaml`: DeepSpeed ZeRO-3 configuration
- `.env`: Configuration for API keys and model repository
- `requirements.txt`: Python dependencies
- `setup.sh`: Environment setup script
- `logs/`: Directory for training logs and checkpoints

## Monitoring Training
- Tmux session: `tmux attach -t granite_training_*`
- Real-time logs: `tail -f logs/train_logs_[timestamp].out`
- Training metrics: Visit your Weights & Biases dashboard
- Checkpoint status: Check `logs/checkpoints.log`
- Model checkpoints: Automatically saved to Hugging Face Hub

## Training Recovery
The training process now includes automatic recovery mechanisms:
- Automatic checkpoint detection and resumption
- Up to 3 retry attempts on failure
- Comprehensive logging of training state
- Safe tmux session management

## License & Attribution
This project is licensed under the [Apache 2.0 License](LICENSE).
The Granite 3.1-2b model used in this project is developed and provided by IBM.

## Safety Features
- Tmux session management for reliable training
- Automatic recovery from failures
- Comprehensive logging and monitoring
- Secure API key management through `.env`
- Automated model backup to Hugging Face Hub
- Structured error handling and reporting