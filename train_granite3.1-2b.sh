#!/bin/bash
set -e

# Configuration
SESSION_NAME="granite_training_$(date +%Y%m%d_%H%M%S)"
MAX_RETRIES=3
RETRY_COUNT=0
MODEL_DIR="outputs/granite-3.1-2b-GRPO"
LOG_DIR="logs"

# Setup logging
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_logs_$(date +%Y%m%d_%H%M%S).out"
CHECKPOINT_LOG="${LOG_DIR}/checkpoints.log"

# Function to find latest checkpoint
find_latest_checkpoint() {
    ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1
}

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to handle cleanup
cleanup() {
    local exit_status=$1
    
    log_message "Cleaning up with exit status: $exit_status"
    
    if [ $exit_status -eq 0 ] && [ -d "$MODEL_DIR" ]; then
        log_message "Training completed successfully. Uploading model..."
        
        if [ -n "$HF_TOKEN" ]; then
            python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$MODEL_DIR',
    repo_id='$HF_REPO_ID',
    token='$HF_TOKEN'
)
"
            log_message "Model upload completed"
        else
            log_message "WARNING: HF_TOKEN not set, skipping model upload"
        fi
    else
        log_message "Training failed or model directory not found"
    fi
}

# Function to start training
start_training() {
    local resume_from=""
    local latest_checkpoint=$(find_latest_checkpoint)
    
    if [ -n "$latest_checkpoint" ]; then
        resume_from="--resume_from_checkpoint $latest_checkpoint"
        log_message "Resuming from checkpoint: $latest_checkpoint"
    fi

    # Create new tmux session
    tmux new-session -d -s "$SESSION_NAME" "
        # Load environment
        source venv/bin/activate
        
        # Export variables
        export WANDB_API_KEY='$WANDB_API_KEY'
        
        # Start training
        accelerate launch --num_processes 7 --config_file src/zero3.yaml src/train_gsm8k.py \
            --output_dir $MODEL_DIR \
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
            --run_name 'granite-3.1-2b-GRPO-gsm8k-8gpu' \
            $resume_from \
            2>&1 | tee -a $LOG_FILE
        
        # Record exit status
        echo \$? > ${LOG_DIR}/exit_status
    "
}

# Function to check training status
check_training_status() {
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        if [ -f "${LOG_DIR}/exit_status" ]; then
            local status=$(cat "${LOG_DIR}/exit_status")
            return $status
        fi
        return 1
    fi
    return 0
}

# Main execution
log_message "Starting training script"

# Load environment variables
if [ -f .env ]; then
    source .env
    log_message "Loaded environment variables"
else
    log_message "Error: .env file not found"
    exit 1
fi

# Main training loop with retries
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    start_training
    log_message "Training started in tmux session: $SESSION_NAME"
    
    # Monitor training
    while true; do
        sleep 300  # Check every 5 minutes
        if ! check_training_status; then
            exit_status=$(cat "${LOG_DIR}/exit_status" 2>/dev/null || echo "1")
            
            if [ "$exit_status" = "0" ]; then
                log_message "Training completed successfully"
                cleanup 0
                exit 0
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                log_message "Training failed. Attempt $RETRY_COUNT of $MAX_RETRIES"
                break
            fi
        fi
    done
done

log_message "Maximum retries reached. Training failed."
cleanup 1
exit 1