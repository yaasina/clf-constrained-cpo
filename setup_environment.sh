#!/bin/bash
echo "Setting up clf-cpo environment for Unix-based systems..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda and try again."
    exit 1
fi

# Create environment from environment.yml
echo "Creating Conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
eval "$(conda shell.bash hook)"
conda activate clf-cpo

# Verify the installation
echo "Verifying installation..."
python -c "import torch; import pytorch_lightning; import wandb; import hydra; import plotly; print('All required packages are installed successfully!')"

# If no errors, finish setup
if [ $? -eq 0 ]; then
    echo "Setup completed successfully!"
    echo "To activate the environment, run: conda activate clf-cpo"
else
    echo "Setup encountered issues. Please check the error messages above."
fi

# Make sure scripts are executable
chmod +x *.py 2>/dev/null || true

echo "Environment is ready! You can now run experiments."