#!/bin/bash
set -e
echo "=== Starting RAGBench training run ==="

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

# Function to handle cleanup and status recording
cleanup_and_shutdown() {
    local exit_status=$1
    local model_path="outputs/granite-3.1-2b-ragbench"
    
    echo "[$(date)] Processing completion with exit status: $exit_status"
    
    if [ $exit_status -eq 0 ] && [ -d "$model_path" ]; then
        echo "[$(date)] Training completed successfully. Uploading model..."
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
            echo "[$(date)] Model upload completed"
        fi
    else
        echo "[$(date)] Training failed or model directory not found. Exit status: $exit_status"
        echo "[$(date)] Model directory exists: $([ -d "$model_path" ] && echo "Yes" || echo "No")"
    fi
    
    # Create a status file for monitoring
    echo "Status: ${exit_status}" > training_status.txt
    echo "Completed at: $(date)" >> training_status.txt
}

# Create output directory
mkdir -p outputs/granite-3.1-2b-ragbench

# Main training command
(nohup accelerate launch --num_processes 7 --config_file src/zero3.yaml src/train_ragbench.py \
--output_dir outputs/granite-3.1-2b-ragbench \
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
--run_name "granite-3.1-2b-ragbench" \
> "$log_file" 2>&1) || { exit_status=$?; cleanup_and_shutdown $exit_status; exit $exit_status; }

# If we get here, training completed successfully
cleanup_and_shutdown 0