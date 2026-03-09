# Neural Control Lyapunov Function (Neural CLF)

A Python framework for learning control-affine dynamics models and control Lyapunov functions for nonlinear systems using PyTorch Lightning and Hydra.

## Project Overview

This project implements a two-stage learning approach for nonlinear control systems:

1. **Dynamics Learning**: Learn control-affine dynamics models using neural networks in an ensemble fashion to capture uncertainty
2. **CLF Learning**: Learn a Control Lyapunov Function (CLF) with a QP-based controller to stabilize the learned dynamics

The framework is designed to work with continuous control environments from Gymnasium and can be extended to custom environments.

## Features

- PyTorch Lightning implementation for better code organization and reproducibility
- Hydra configuration for flexible experiment setup
- Weights & Biases integration for experiment tracking and visualization
- Ensemble-based learning of control-affine dynamics
- Uncertainty quantification through model ensembles
- Neural network-based CLF representation
- QP-based optimal control policy with cvxpylayers for differentiable optimization
- Comprehensive visualization suite with Plotly
- Support for various environments

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda env create -f environment.yml
conda activate clf-cpo

# Or manually:
conda create -n clf-cpo python=3.9
conda activate clf-cpo

# Install PyTorch (adjust for your CUDA version if using GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install gymnasium matplotlib cvxpy cvxpylayers pytorch-lightning hydra-core omegaconf wandb plotly
```

### Option 2: Using venv

```bash
# Create a new virtual environment
python -m venv clf_env
source clf_env/bin/activate  # On Windows: clf_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using the provided scripts

```bash
# On Windows
setup_environment.bat

# On Linux/Mac
chmod +x setup_environment.sh
./setup_environment.sh
```

## Usage with Hydra

The project uses Hydra for configuration management, making it easy to customize experiments.

### Running Experiments

The main entry point is `src/training.py`, which can be run with different Hydra configurations.

#### Basic Usage

```bash
# Run with default parameters (defined in configs/config.yaml)
python src/training.py

# Override specific parameters
python src/training.py experiment.task=dynamics_learning
python src/training.py experiment.task=clf_training

# Run the full pipeline (dynamics learning + CLF training)
python src/training.py experiment.task=full_pipeline
```

#### Advanced Configuration

```bash
# Specify a different model configuration
python src/training.py model=dynamics_ensemble

# Change training parameters
python src/training.py training.max_epochs=200 training.batch_size=64

# Configure optimizer
python src/training.py optimizer=adam optimizer.lr=0.001

# Override several parameters at once
python src/training.py \
  experiment.task=full_pipeline \
  model=dynamics_ensemble \
  training.max_epochs=200 \
  optimizer.lr=0.001 \
  training.batch_size=64
```

#### Experiment Configuration

For dynamics learning experiments:
```bash
python src/training.py \
  experiment.task=dynamics_learning \
  model=dynamics_ensemble \
  experiment.dynamics.ensemble_size=5 \
  experiment.dynamics.hidden_dim=128
```

For CLF learning experiments:
```bash
python src/training.py \
  experiment.task=clf_training \
  model=clf_network \
  experiment.clf.hidden_dim=128 \
  experiment.dynamics_model_path=/path/to/trained/dynamics/model.ckpt
```

#### Multi-run Experiments

Hydra supports running multiple configurations with a single command:

```bash
# Run with several learning rates
python src/training.py -m optimizer.lr=0.001,0.0005,0.0001

# Run with different models and seeds
python src/training.py -m model=clf_network,dynamics_ensemble seed=42,123,456
```

### Configuration Structure

The project uses a hierarchical configuration structure:

- `configs/config.yaml`: Top-level configuration with defaults
- `configs/experiment/*.yaml`: Experiment-specific configurations
- `configs/model/*.yaml`: Model architectures
- `configs/optimizer/*.yaml`: Optimizer configurations
- `configs/logger/*.yaml`: Logging configurations

## Project Structure

```
clf-constrained-cpo/
├── configs/               # Hydra configuration files
│   ├── config.yaml        # Main configuration
│   ├── experiment/        # Experiment configurations
│   ├── logger/            # Logger configurations
│   ├── model/             # Model configurations
│   └── optimizer/         # Optimizer configurations
├── src/                   # Source code
│   ├── training.py        # Main training script
│   ├── data/              # Data handling
│   │   └── data_module.py # PyTorch Lightning data modules
│   ├── models/            # Model implementations
│   │   ├── clf.py         # CLF network implementation
│   │   └── dynamics.py    # Dynamics model implementation
│   ├── solvers/           # Optimization solvers
│   │   └── clf_qp_solver.py # QP solver for CLF
│   └── utils/             # Utility functions
├── data_collection.py     # Data collection from environments
├── pendulum_dynamics.py   # Pendulum environment implementation
├── pendulum_utils.py      # Pendulum utilities
├── environment.yml        # Conda environment file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Visualizing Results with Weights & Biases

The project integrates with Weights & Biases (wandb) for experiment tracking and visualization:

1. **Setup a W&B account**:
   ```bash
   pip install wandb
   wandb login
   ```

2. **Run an experiment**:
   ```bash
   python src/training.py
   ```

3. **View results**:
   - Open the URL provided in the console
   - Or go to https://wandb.ai/your-username/your-project

W&B automatically tracks:
- Training metrics
- Validation metrics
- Model parameters
- Hyperparameters
- CLF visualizations
- Lie derivative visualizations
- QP solver results
- Admissible control sets

## Environment Integration

To use your own environments:

1. Create a custom data collection script based on `data_collection.py`
2. Generate data in the format expected by the data modules:
   - States: `[batch_size, state_dim]`
   - Actions: `[batch_size, action_dim]`
   - Next states: `[batch_size, state_dim]`
   - (Optional) State derivatives: `[batch_size, state_dim]`

3. Configure your experiment:
   ```yaml
   # configs/experiment/custom_env.yaml
   experiment:
     task: full_pipeline
     env_name: CustomEnv
     state_dim: <your_state_dim>
     action_dim: <your_action_dim>
     data_path: /path/to/your/data.pt
   ```

## Mathematical Background

For a detailed understanding of the mathematical foundations of this project, refer to `dynamics_learning_explanation.md` which covers:
- Control affine systems representation
- Neural network design for dynamics learning
- Ensemble methods for uncertainty quantification
- CLF theory and implementation details

## Troubleshooting

### Common Issues

1. **CUDA Memory Issues**:
   ```bash
   # Reduce batch size
   python src/training.py training.batch_size=32
   
   # Use lower precision
   python src/training.py device.precision=16
   ```

2. **QP Solver Failures**:
   ```bash
   # Increase relaxation weight
   python src/training.py experiment.clf.qp_solver.lambda_param=5.0
   
   # Increase max retries
   python src/training.py experiment.clf.qp_solver.max_retries=5
   ```

3. **Wandb Connection Issues**:
   ```bash
   # Run in offline mode
   python src/training.py logger.offline=true
   
   # Later, sync the results
   wandb sync ./wandb/run-*
   ```

### Getting Help

If you encounter issues not covered here, please check:
- The documentation in the source code
- The mathematical background in `dynamics_learning_explanation.md`
- The PyTorch Lightning documentation: https://lightning.ai/docs/pytorch/stable/
- The Hydra documentation: https://hydra.cc/docs/intro/

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:
```
@software{neural_clf,
  author = {Author},
  title = {Neural Control Lyapunov Functions for Nonlinear Control},
  year = {2023},
  url = {https://github.com/username/neural_clf}
}
```