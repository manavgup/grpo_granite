#!/bin/bash
set -e
echo "=== Starting the training run ==="

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Export WandB API key
export WANDB_API_KEY="$WANDB_API_KEY"

source venv/bin/activate
ts=$(date +%Y%m%d_%H%M%S)
log_file="train_logs_${ts}.out"

# Function to handle cleanup and shutdown
cleanup_and_shutdown() {
    local exit_status=$1
    local model_path="outputs/granite-3.1-2b-GRPO"
    
    if [ $exit_status -eq 0 ]; then
        echo "Training completed successfully. Uploading model..."
        # Upload to Hugging Face (assuming HF_TOKEN is in .env)
        if [ -n "$HF_TOKEN" ]; then
            python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$model_path',
    repo_id='$HF_REPO_ID',
    token='$HF_TOKEN'
)
"
        fi
    else
        echo "Training failed with exit status $exit_status"
    fi

    # Shutdown the server
    echo "Initiating server shutdown..."
    sudo shutdown -h now
}

# Main training command
(nohup accelerate launch --num_processes 8 --config_file src/zero3.yaml src/train_gsm8k.py \
--output_dir outputs/granite-3.1-2b-GRPO \
--model_name_or_path ibm-granite/granite-3.1-2b-instruct \
--max_prompt_length 2048 \
--max_completion_length 2048 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 3e-6 \
--adam_beta1 0.9 \
--adam_beta2 0.99 \
--weight_decay 0.1 \
--warmup_ratio 0.1 \
--logging_steps 1 \
--num_generations 3 \
--save_steps 50 \
--max_steps 1000 \
--torch_dtype bfloat16 \
--use_vllm \
--vllm_gpu_memory_utilization 0.7 \
--bf16 \
--report_to wandb \
--run_name "granite-3.1-2b-GRPO-gsm8k-8gpu" \
> "$log_file" 2>&1) || cleanup_and_shutdown $?

# If we get here, training completed successfully
cleanup_and_shutdown 0