# GRPO Granite

This project implements a Group Relative Policy Optimization (GRPO) approach to fine-tune IBM's Granite 3.1-2b language model on the GSM8K dataset for enhanced mathematical reasoning capabilities.

## Key Features

- GRPO implementation with multiple reward functions:
  - Answer accuracy
  - Number formatting
  - Reasoning structure
- Integration with IBM Granite 3.1-2b model
- DeepSpeed ZeRO-3 optimization for efficient training
- VLLM acceleration for faster model inference
- Custom reward functions for evaluating model outputs
- Automatic model upload to Hugging Face Hub
- Automated server shutdown after training
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
- Hardware requirements:
  - 8x GPUs with at least 24GB of memory each
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
1. Load environment variables from `.env`
2. Initialize Weights & Biases logging
3. Train the model using DeepSpeed ZeRO-3 optimization
4. Upload the trained model to Hugging Face Hub
5. Automatically shut down the server upon completion

Training logs will be saved in `train_logs_[timestamp].out` files.

## Project Structure

- `src/`
  - `train_granite3.1-2b.sh`: Training script with auto-shutdown
  - `train_gsm8k.py`: Main training implementation
  - `zero3.yaml`: DeepSpeed ZeRO-3 configuration
- `.env`: Configuration for API keys and model repository
- `requirements.txt`: Python dependencies
- `setup.sh`: Environment setup script

## Monitoring Training

- Real-time logs: `tail -f train_logs_[timestamp].out`
- Training metrics: Visit your Weights & Biases dashboard
- Model checkpoints: Automatically saved to Hugging Face Hub

## License & Attribution

This project is licensed under the [Apache 2.0 License](LICENSE).

The Granite 3.1-2b model used in this project is developed and provided by IBM.

## Safety Features

- Automatic server shutdown after training completion or failure
- Secure API key management through `.env` file
- Automated model backup to Hugging Face Hub before shutdown