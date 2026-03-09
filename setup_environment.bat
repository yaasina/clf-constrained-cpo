@echo off
echo Setting up clf-cpo environment for Windows...

:: Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda not found in PATH. Please install Miniconda or Anaconda and try again.
    exit /b 1
)

:: Create environment from environment.yml
echo Creating Conda environment from environment.yml...
call conda env create -f environment.yml

:: Activate environment
call conda activate clf-cpo

:: Verify the installation
echo Verifying installation...
python -c "import torch; import pytorch_lightning; import wandb; import hydra; import plotly; print('All required packages are installed successfully!')"

:: If no errors, finish setup
if %ERRORLEVEL% equ 0 (
    echo Setup completed successfully!
    echo To activate the environment, run: conda activate clf-cpo
) else (
    echo Setup encountered issues. Please check the error messages above.
)

:: Return to the main directory
cd %~dp0
echo Environment is ready! You can now run experiments.