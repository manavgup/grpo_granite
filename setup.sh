#!/bin/bash
set -e

PROJECT_NAME="my_project"  # Change this to your actual project name
PROJECT_DIR="$HOME/$PROJECT_NAME"

echo "=== Setting up the project directory ==="
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "=== Setting up the Python environment ==="
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements if available
echo "=== Installing requirements ==="
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping package installation."
fi

# Optionally install a custom package (uncomment and modify if needed)
# echo "=== Installing a custom package ==="
# pip install some-custom-package

# Prompt for Hugging Face API key (optional)
echo "=== Setting up Hugging Face API key (Optional) ==="
read -rp "Enter your Hugging Face API key (or press Enter to skip): " HF_TOKEN
if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN
    echo "export HF_TOKEN=$HF_TOKEN" >> venv/bin/activate
    echo "Hugging Face API key has been set."
else
    echo "Skipping Hugging Face API key setup."
fi

# Prompt for WandB API key (optional)
echo "=== Setting up WandB API key (Optional) ==="
read -rp "Enter your WandB API key (or press Enter to skip): " WANDB_API_KEY
if [[ -n "$WANDB_API_KEY" ]]; then
    export WANDB_API_KEY
    echo "export WANDB_API_KEY=$WANDB_API_KEY" >> venv/bin/activate
    echo "WandB API key has been set."
else
    echo "Skipping WandB API key setup."
fi

echo "=== Setup complete! ==="
echo "To activate the virtual environment, run: source venv/bin/activate"
