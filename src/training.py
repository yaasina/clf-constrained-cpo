"""
Main training module for dynamics and CLF models using PyTorch Lightning and Hydra.
"""

import os
import sys
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Add the project root to the path to make imports work from any location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Update imports to be relative to the project root
from models.dynamics import ControlAffineNetworkLightning, DynamicsEnsembleLightning
from models.clf import CLFNetworkLightning
from solvers.clf_qp_solver import CLFQPSolverLightning
from data.data_module import DynamicsDataModule, CLFDataModule

# Import gym for environment handling
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

def get_space_dimension(space):
    """
    Get the dimension of a gym space.
    
    Args:
        space: Gym space (Box, Discrete, etc.)
        
    Returns:
        Dimension of the space
    """
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    pl.seed_everything(seed)  # covers torch, numpy, and python random


def setup_callbacks(config: DictConfig) -> List[pl.Callback]:
    """
    Set up callbacks for PyTorch Lightning.
    
    Args:
        config: Hydra configuration
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        save_top_k=config.checkpoint.save_top_k,
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        auto_insert_metric_name=config.checkpoint.auto_insert_metric_name,
        every_n_epochs=config.checkpoint.every_n_epochs,
        save_last=config.checkpoint.save_last
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config.training.early_stopping.enabled:
        early_stopping_callback = EarlyStopping(
            monitor=config.training.early_stopping.monitor,
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            mode=config.training.early_stopping.mode
        )
        callbacks.append(early_stopping_callback)
    
    return callbacks


def train_dynamics_model(config: DictConfig) -> Tuple[pl.LightningModule, Dict[str, Any]]:
    """
    Train a dynamics model using PyTorch Lightning.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Trained dynamics model and training results
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize logger
    logger = WandbLogger(
        project=config.logger.project,
        name=config.logger.name,
        save_dir=config.logger.save_dir,
        log_model=config.logger.log_model,
        notes=config.logger.notes,
        tags=config.logger.tags,
        group=config.logger.group
    )
    
    # Initialize environment if using environment data
    env = None
    if hasattr(config.experiment, 'env_name') and config.experiment.env_name is not None:
        from data_collection import collect_trajectory, process_trajectory
        
        print(f"Setting up environment: {config.experiment.env_name}")
        env = gym.make(config.experiment.env_name)
    
    # Get configuration parameters
    num_trajectories = config.experiment.data.get('num_trajectories', 20)
    traj_length = config.experiment.data.get('traj_length', 200)
    buffer_size = config.experiment.data.get('buffer_size', num_trajectories)  # Default to keeping all trajectories
    epochs_per_trajectory = config.experiment.data.get('epochs_per_trajectory', 5)  # New parameter
    batch_size = config.experiment.training.get("batch_size", config.training.batch_size)
    
    # Calculate log_every_n_steps based on the number of batches and epochs
    # 4 batches per epoch is a typical default minimum for logging
    log_every_n_steps = config.experiment.training.get("log_every_n_steps", 
                                                      max(1, epochs_per_trajectory // 2))
    
    # Configure model dimensions - will be set after first trajectory collection
    state_dim = None
    action_dim = None
    model = None
    
    # Initialize trajectory buffer
    trajectory_buffer = []
    
    # Initialize callbacks
    callbacks = setup_callbacks(config)
    
    # Initialize data collection counter 
    total_epochs_trained = 0
    
    # Train incrementally with each trajectory
    for traj_idx in range(num_trajectories):
        print(f"\n=== Collecting trajectory {traj_idx+1}/{num_trajectories} ===")
        
        if env is not None:
            # Collect a new trajectory
            new_trajectory = collect_trajectory(
                env, 
                length=traj_length, 
                random_seed=config.seed + traj_idx
            )
            
            # Add to buffer using FIFO if buffer size is limited
            trajectory_buffer.append(new_trajectory)
            if len(trajectory_buffer) > buffer_size:
                trajectory_buffer.pop(0)  # Remove oldest trajectory
            
            # Process all trajectories in buffer into a dataset
            states, actions, next_states = process_trajectory(trajectory_buffer)
            print(f"Dataset size after trajectory {traj_idx+1}: {states.shape[0]} samples")
        else:
            # Load data from file if specified instead of collecting from environment
            if hasattr(config.experiment.data, "data_path") and config.experiment.data.data_path:
                # Load data code here (implementation depends on your data format)
                # This is a placeholder - modify according to your data loading logic
                print(f"Loading data from {config.experiment.data.data_path}")
                # states, actions, next_states = load_data(config.experiment.data.data_path)
                raise NotImplementedError("Data loading from files is not implemented yet")
            else:
                raise ValueError("Neither environment nor data path specified for training")
        
        # Create data module (direct instantiation - avoids _target_ string issues) (L4)
        data_module = DynamicsDataModule(
            batch_size=batch_size,
            num_workers=config.experiment.training.get("num_workers", 4),
            normalize=config.experiment.dynamics.get("normalize_data", True),
            random_seed=config.seed,
            states=states,
            actions=actions,
            next_states=next_states,
            persistent_workers=config.experiment.data.get("persistent_workers", False),
            train_ratio=config.experiment.data.get("train_ratio", 0.8),
            val_ratio=config.experiment.data.get("val_ratio", 0.1),
            test_ratio=config.experiment.data.get("test_ratio", 0.1),
        )
        data_module.prepare_data()
        data_module.setup()
        
        # Initialize model if first trajectory
        if model is None:
            if env is not None:
                # Use environment spaces to determine dimensions
                state_dim = get_space_dimension(env.observation_space)
                action_dim = get_space_dimension(env.action_space)
            else:
                # Get a batch to determine dimensions
                batch = next(iter(data_module.train_dataloader()))
                state_dim = batch["states"].shape[1]
                action_dim = batch["actions"].shape[1]
            
            # Update the config with the inferred dimensions
            config_dict = OmegaConf.to_container(config.model, resolve=True)
            config_dict.update({"state_dim": state_dim, "action_dim": action_dim})
            model_config = OmegaConf.create(config_dict)
            
            # Initialize model
            model = hydra.utils.instantiate(model_config)
            
            # Log the inferred dimensions
            logger.experiment.config.update({
                "state_dim": state_dim,
                "action_dim": action_dim,
                "epochs_per_trajectory": epochs_per_trajectory,
                "buffer_size": buffer_size,
            })
            print(f"Inferred dimensions from data: state_dim={state_dim}, action_dim={action_dim}")
        
        # Configure trainer for this trajectory
        # Determine if sanity checks should be run
        num_sanity_val_steps = config.experiment.training.get("num_sanity_val_steps", 0 if traj_idx > 0 else 1)
        
        # Configure trainer for this trajectory
        trainer = pl.Trainer(
            accelerator=config.device.accelerator,
            devices=config.device.devices,
            max_epochs=total_epochs_trained + epochs_per_trajectory,
            min_epochs=total_epochs_trained + 1,  # Ensure at least one epoch of training
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=config.training.gradient_clip_val,
            val_check_interval=config.training.val_check_interval,
            precision=config.device.precision,
            log_every_n_steps=log_every_n_steps,  # Adaptive logging
            num_sanity_val_steps=num_sanity_val_steps,  # Configurable sanity check
        )
        
        # Train model on current dataset
        print(f"Training for {epochs_per_trajectory} epochs on trajectory {traj_idx+1}")
        trainer.fit(model, data_module)
        
        # Update total epochs trained
        total_epochs_trained += epochs_per_trajectory

        # Log trajectory completion
        if isinstance(logger, pl.loggers.WandbLogger):
            logger.experiment.log({
                "trajectory_completed": traj_idx + 1,
                "total_epochs_trained": total_epochs_trained
            })
    
    # Test model on final dataset
    print("\n=== Testing final model ===")
    test_results = trainer.test(model, data_module)
    
    # Log best model path
    best_model_path = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
    
    print(f"Best model path: {best_model_path}")
    logger.experiment.summary["best_model_path"] = best_model_path
    
    # Cleanup
    if env is not None:
        env.close()
    
    return model, {"test_results": test_results, "best_model_path": best_model_path}


def train_clf_model(
    config: DictConfig, 
    dynamics_model: Optional[pl.LightningModule] = None
) -> Tuple[pl.LightningModule, Dict[str, Any]]:
    """
    Train a CLF model using PyTorch Lightning.
    
    Args:
        config: Hydra configuration
        dynamics_model: Pre-trained dynamics model for Lie derivative computation
        
    Returns:
        Trained CLF model and training results
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize logger
    logger = WandbLogger(
        project=config.logger.project,
        name=config.logger.name,
        save_dir=config.logger.save_dir,
        log_model=config.logger.log_model,
        notes=config.logger.notes,
        tags=config.logger.tags,
        group=config.logger.group
    )
    
    # Initialize environment to collect states
    env = None
    states = None
    if hasattr(config.experiment, 'env_name') and config.experiment.env_name is not None:
        from data_collection import collect_trajectory, process_trajectory
        
        print(f"Setting up environment for CLF training: {config.experiment.env_name}")
        env = gym.make(config.experiment.env_name)
        
        # Collect trajectories to generate state samples for CLF training
        num_trajectories = config.experiment.data.get('num_trajectories', 5)
        traj_length = config.experiment.data.get('traj_length', 50)
        
        trajectory_buffer = []
        
        # Collect trajectories
        for traj_idx in range(num_trajectories):
            print(f"\n=== Collecting trajectory {traj_idx+1}/{num_trajectories} for CLF training ===")
            
            # Collect a new trajectory
            new_trajectory = collect_trajectory(
                env, 
                length=traj_length, 
                random_seed=config.seed + traj_idx
            )
            
            trajectory_buffer.append(new_trajectory)
        
        # Process trajectories to get states
        states, _, _ = process_trajectory(trajectory_buffer)
        print(f"Generated dataset for CLF training: {states.shape[0]} state samples")
        
        # Clean up environment
        env.close()
    
    # Direct instantiation — avoids _target_ string issues (L4)
    data_module = CLFDataModule(
        states=states,
        batch_size=config.experiment.training.get("batch_size", config.training.batch_size),
        num_workers=config.experiment.training.get("num_workers", 4),
        normalize=config.experiment.dynamics.get("normalize_data", True),
        random_seed=config.seed,
        persistent_workers=config.experiment.data.get("persistent_workers", False),
        train_ratio=config.experiment.data.get("train_ratio", 0.8),
        val_ratio=config.experiment.data.get("val_ratio", 0.1),
        test_ratio=config.experiment.data.get("test_ratio", 0.1),
    )
    if hasattr(config.experiment.data, "data_path") and config.experiment.data.data_path:
        data_module.data_path = config.experiment.data.data_path
    
    # Setup the data module to access the dimensions
    data_module.prepare_data()
    data_module.setup()
    
    # Extract dimensions from the data
    # Get a batch from the training dataloader to determine dimensions
    batch = next(iter(data_module.train_dataloader()))
    state_dim = batch["states"].shape[1]
    
    # Also get action_dim from dynamics model if available
    action_dim = None
    if dynamics_model is not None and hasattr(dynamics_model, 'action_dim'):
        action_dim = dynamics_model.action_dim
    
    # Update the config with the inferred dimensions
    config_dict = OmegaConf.to_container(config.model, resolve=True)
    config_dict.update({"state_dim": state_dim})
    if action_dim is not None:
        config_dict.update({"action_dim": action_dim})
    model_config = OmegaConf.create(config_dict)
    
    # Initialize model with dynamic dimensions
    model = hydra.utils.instantiate(model_config)

    # Attach dynamics model so training_step can access it without putting it in batches (C2)
    model.dynamics_model = dynamics_model

    # Attach QP solver so training_step can use the full self-supervised CLF loss (C5)
    if action_dim is not None:
        model.qp_solver = CLFQPSolverLightning(action_dim=action_dim)

    # Set pendulum equilibrium on the model buffer so CLF loss targets the right point (L6)
    if hasattr(model, 'equilibrium'):
        try:
            from pendulum_utils import get_pendulum_equilibrium
            model.equilibrium.copy_(get_pendulum_equilibrium())
        except Exception:
            pass

    # Log the inferred dimensions
    logger.experiment.config.update({
        "state_dim": state_dim
    })
    if action_dim is not None:
        logger.experiment.config.update({"action_dim": action_dim})
    
    print(f"Inferred dimensions from data: state_dim={state_dim}" + 
          (f", action_dim={action_dim}" if action_dim is not None else ""))
    
    # Set up callbacks
    callbacks = setup_callbacks(config)
    
    # Get sanity check configuration
    num_sanity_val_steps = config.experiment.training.get("num_sanity_val_steps", 1)
    
    # Configure trainer
    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        devices=config.device.devices,
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        val_check_interval=config.training.val_check_interval,
        precision=config.device.precision,
        num_sanity_val_steps=num_sanity_val_steps
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    test_results = trainer.test(model, data_module)
    
    # Log best model path
    best_model_path = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
    
    print(f"Best model path: {best_model_path}")
    logger.experiment.summary["best_model_path"] = best_model_path
    
    return model, {"test_results": test_results, "best_model_path": best_model_path}


def evaluate_control_policy(
    config: DictConfig, 
    clf_model: pl.LightningModule, 
    dynamics_model: pl.LightningModule
) -> Dict[str, Any]:
    """
    Evaluate the CLF-QP control policy.
    
    Args:
        config: Hydra configuration
        clf_model: Trained CLF model
        dynamics_model: Trained dynamics model
        
    Returns:
        Evaluation results
    """
    # Initialize QP solver
    qp_solver = hydra.utils.instantiate(
        config.experiment.clf.qp_solver,
        _target_=CLFQPSolverLightning
    )
    
    # Initialize wandb logger for visualizations
    logger = WandbLogger(
        project=config.logger.project,
        name=f"{config.logger.name}_eval",
        save_dir=config.logger.save_dir,
        log_model=False,
        notes=f"Evaluation of {config.logger.name}",
        tags=config.logger.tags + ["evaluation"],
        group=config.logger.group
    )
    
    qp_solver.logger = logger
    
    # Set up test states (grid or samples)
    if hasattr(config.experiment, "eval_grid") and config.experiment.eval_grid:
        # Create a grid of test states (for 2D visualization)
        x_range = torch.linspace(-3, 3, 20)
        y_range = torch.linspace(-3, 3, 20)
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
        states = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        if dynamics_model.state_dim > 2:
            # Pad with zeros for higher-dimensional states
            zeros = torch.zeros(len(states), dynamics_model.state_dim - 2)
            states = torch.cat([states, zeros], dim=1)
    else:
        # Sample random states
        n_samples = config.experiment.get("n_eval_samples", 1000)
        states = torch.randn(n_samples, dynamics_model.state_dim) * 2.0
    
    # Move to same device as models
    device = next(clf_model.parameters()).device
    states = states.to(device)
    
    # Solve QP for each state
    results = qp_solver.solve_batch(
        states, clf_model, dynamics_model, 
        batch_size=config.training.batch_size,
        log_results=True
    )
    
    # For a few states, compute and visualize the admissible control set
    if hasattr(config.experiment, "visualize_admissible_controls") and config.experiment.visualize_admissible_controls:
        # Sample a few states at different distances from equilibrium
        state_norms = torch.norm(states, dim=1)
        
        # Get indices of states at different norm ranges
        close_idx = torch.where((state_norms > 0.5) & (state_norms < 1.0))[0]
        mid_idx = torch.where((state_norms > 1.5) & (state_norms < 2.0))[0]
        far_idx = torch.where(state_norms > 2.5)[0]
        
        # Sample one state from each range if available
        sample_states = []
        if len(close_idx) > 0:
            sample_states.append(states[close_idx[0]])
        if len(mid_idx) > 0:
            sample_states.append(states[mid_idx[0]])
        if len(far_idx) > 0:
            sample_states.append(states[far_idx[0]])
        
        # Compute admissible control sets
        for state in sample_states:
            qp_solver.compute_admissible_control_set(
                state, clf_model, dynamics_model, 
                num_samples=1000, log_results=True
            )
    
    return results


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig) -> Dict[str, Any]:
    """
    Main entry point for training and evaluating dynamics and CLF models.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Dictionary with results
    """
    # Print configuration
    print(OmegaConf.to_yaml(config))
    
    results = {}
    
    # Train dynamics model
    if config.experiment.task == "dynamics_learning":
        model, train_results = train_dynamics_model(config)
        results["dynamics"] = train_results
        
    # Train CLF model
    elif config.experiment.task == "clf_training":
        # Load pre-trained dynamics model if path is provided
        dynamics_model = None
        if "dynamics_model_path" in config.experiment and config.experiment.dynamics_model_path:
            print(f"Loading dynamics model from {config.experiment.dynamics_model_path}")
            # The model class is determined by the checkpoint content
            dynamics_model = DynamicsEnsembleLightning.load_from_checkpoint(
                config.experiment.dynamics_model_path
            )
            dynamics_model.eval()
        
        # Train CLF model
        clf_model, train_results = train_clf_model(config, dynamics_model)
        results["clf"] = train_results
        
        # Evaluate control policy if requested
        if dynamics_model is not None and config.experiment.get("evaluate_control", False):
            eval_results = evaluate_control_policy(config, clf_model, dynamics_model)
            results["control_evaluation"] = eval_results
    
    # Run full pipeline - alternating dynamics and CLF training
    elif config.experiment.task == "full_pipeline":
        # Initialize environment if using environment data
        env = None
        if hasattr(config.experiment, 'env_name') and config.experiment.env_name is not None:
            from data_collection import collect_trajectory, process_trajectory
            
            print(f"Setting up environment: {config.experiment.env_name}")
            env = gym.make(config.experiment.env_name)
        else:
            raise ValueError("Environment name must be provided for full pipeline")
        
        # Get configuration parameters
        num_trajectories = config.experiment.data.get('num_trajectories', 5)
        traj_length = config.experiment.data.get('traj_length', 50)
        buffer_size = config.experiment.data.get('buffer_size', 3)
        epochs_per_trajectory = config.experiment.data.get('epochs_per_trajectory', 5)
        batch_size = config.experiment.training.get("batch_size", config.training.batch_size)
        
        # Initialize trajectory buffer
        trajectory_buffer = []
        
        # Initialize models
        dynamics_model = None
        clf_model = None
        
        # Initialize logger for dynamics
        dynamics_logger = WandbLogger(
            project=config.logger.project,
            name=f"{config.logger.name}_dynamics",
            save_dir=config.logger.save_dir,
            log_model=config.logger.log_model,
            notes=f"{config.logger.notes} - Dynamics",
            tags=config.logger.tags + ["dynamics"],
            group=config.logger.group
        )
        
        # Initialize logger for CLF
        clf_logger = WandbLogger(
            project=config.logger.project,
            name=f"{config.logger.name}_clf",
            save_dir=config.logger.save_dir,
            log_model=config.logger.log_model,
            notes=f"{config.logger.notes} - CLF",
            tags=config.logger.tags + ["clf"],
            group=config.logger.group
        )
        
        # Setup callbacks
        dynamics_callbacks = setup_callbacks(config)
        clf_callbacks = setup_callbacks(config)
        
        # Get sanity check configuration
        num_sanity_val_steps = config.experiment.training.get("num_sanity_val_steps", 0)
        
        # Initialize counters
        dynamics_epochs_trained = 0
        clf_epochs_trained = 0
        
        # Train incrementally with each trajectory
        for traj_idx in range(num_trajectories):
            print(f"\n=== Collecting trajectory {traj_idx+1}/{num_trajectories} ===")
            
            # Collect a new trajectory
            new_trajectory = collect_trajectory(
                env, 
                length=traj_length, 
                random_seed=config.seed + traj_idx
            )
            
            # Add to buffer using FIFO if buffer size is limited
            trajectory_buffer.append(new_trajectory)
            if len(trajectory_buffer) > buffer_size:
                trajectory_buffer.pop(0)  # Remove oldest trajectory
            
            # Process all trajectories in buffer into a dataset
            states, actions, next_states = process_trajectory(trajectory_buffer)
            print(f"Dataset size after trajectory {traj_idx+1}: {states.shape[0]} samples")
            
            # ===== STEP 1: Train dynamics model on current trajectory buffer =====
            dynamics_data_module = DynamicsDataModule(
                states=states,
                actions=actions,
                next_states=next_states,
                batch_size=batch_size,
                num_workers=config.experiment.training.get("num_workers", 4),
                normalize=config.experiment.dynamics.get("normalize_data", True),
                random_seed=config.seed,
                persistent_workers=config.experiment.data.get("persistent_workers", False),
                train_ratio=config.experiment.data.get("train_ratio", 0.8),
                val_ratio=config.experiment.data.get("val_ratio", 0.1),
                test_ratio=config.experiment.data.get("test_ratio", 0.1)
            )
            dynamics_data_module.prepare_data()
            dynamics_data_module.setup()
            
            # Initialize dynamics model if first trajectory
            if dynamics_model is None:
                if env is not None:
                    # Use environment spaces to determine dimensions
                    state_dim = get_space_dimension(env.observation_space)
                    action_dim = get_space_dimension(env.action_space)
                else:
                    # Get a batch to determine dimensions
                    batch = next(iter(dynamics_data_module.train_dataloader()))
                    state_dim = batch["states"].shape[1]
                    action_dim = batch["actions"].shape[1]
                
                # Update the config with the inferred dimensions
                dynamics_config = OmegaConf.to_container(config.model, resolve=True)
                dynamics_config.update({"state_dim": state_dim, "action_dim": action_dim})
                dynamics_model_config = OmegaConf.create(dynamics_config)
                
                # Initialize dynamics model
                dynamics_model = hydra.utils.instantiate(dynamics_model_config)
                
                # Log the inferred dimensions
                dynamics_logger.experiment.config.update({
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "epochs_per_trajectory": epochs_per_trajectory,
                    "buffer_size": buffer_size,
                })
                print(f"Inferred dimensions from data: state_dim={state_dim}, action_dim={action_dim}")
            
            # Configure dynamics trainer for this trajectory
            dynamics_trainer = pl.Trainer(
                accelerator=config.device.accelerator,
                devices=config.device.devices,
                max_epochs=dynamics_epochs_trained + epochs_per_trajectory,
                min_epochs=dynamics_epochs_trained + 1,  # Ensure at least one epoch of training
                logger=dynamics_logger,
                callbacks=dynamics_callbacks,
                gradient_clip_val=config.training.gradient_clip_val,
                val_check_interval=config.training.val_check_interval,
                precision=config.device.precision,
                log_every_n_steps=config.experiment.training.get("log_every_n_steps", 2),
                num_sanity_val_steps=num_sanity_val_steps if traj_idx == 0 else 0,
            )
            
            # Train dynamics model on current dataset
            print(f"Training dynamics model for {epochs_per_trajectory} epochs on trajectory {traj_idx+1}")
            dynamics_trainer.fit(dynamics_model, dynamics_data_module)
            
            # Update dynamics epochs counter
            dynamics_epochs_trained += epochs_per_trajectory
            
            # ===== STEP 2: Train CLF model using the updated dynamics model =====
            clf_data_module = CLFDataModule(
                states=states,
                batch_size=batch_size,
                num_workers=config.experiment.training.get("num_workers", 4),
                normalize=config.experiment.dynamics.get("normalize_data", True),
                random_seed=config.seed,
                persistent_workers=config.experiment.data.get("persistent_workers", False),
                train_ratio=config.experiment.data.get("train_ratio", 0.8),
                val_ratio=config.experiment.data.get("val_ratio", 0.1),
                test_ratio=config.experiment.data.get("test_ratio", 0.1)
            )
            clf_data_module.prepare_data()
            clf_data_module.setup()
            
            # Initialize CLF model if first trajectory
            if clf_model is None:
                # Initialize config for CLF model
                clf_config_dict = {
                    "_target_": "src.models.clf.CLFNetworkLightning",
                    "state_dim": state_dim,
                    "hidden_dim": config.experiment.clf.get("hidden_dim", 64),
                    "dropout_rate": 0.1,
                    "learning_rate": 0.001
                }
                clf_model_config = OmegaConf.create(clf_config_dict)
                
                # Initialize CLF model
                clf_model = hydra.utils.instantiate(clf_model_config)

                # Attach dynamics model so training_step can access it (C2)
                clf_model.dynamics_model = dynamics_model

                # Attach QP solver for full self-supervised CLF loss (C5)
                clf_model.qp_solver = CLFQPSolverLightning(action_dim=action_dim)

                # Set equilibrium on model buffer (L6)
                if hasattr(clf_model, 'equilibrium'):
                    try:
                        from pendulum_utils import get_pendulum_equilibrium
                        clf_model.equilibrium.copy_(get_pendulum_equilibrium())
                    except Exception:
                        pass

                # Log CLF model config
                clf_logger.experiment.config.update(clf_config_dict)
            
            # Configure CLF trainer for this trajectory
            clf_trainer = pl.Trainer(
                accelerator=config.device.accelerator,
                devices=config.device.devices,
                max_epochs=clf_epochs_trained + epochs_per_trajectory,
                min_epochs=clf_epochs_trained + 1,  # Ensure at least one epoch of training
                logger=clf_logger,
                callbacks=clf_callbacks,
                gradient_clip_val=config.training.gradient_clip_val,
                val_check_interval=config.training.val_check_interval,
                precision=config.device.precision,
                log_every_n_steps=config.experiment.training.get("log_every_n_steps", 2),
                num_sanity_val_steps=num_sanity_val_steps if traj_idx == 0 else 0,
            )
            
            # Train CLF model using the current dataset and updated dynamics model
            print(f"Training CLF model for {epochs_per_trajectory} epochs on trajectory {traj_idx+1}")
            clf_trainer.fit(clf_model, clf_data_module)
            
            # Update CLF epochs counter
            clf_epochs_trained += epochs_per_trajectory
            
            # Log trajectory completion
            dynamics_logger.experiment.log({
                "trajectory_completed": traj_idx + 1,
                "total_epochs_trained": dynamics_epochs_trained
            })
            clf_logger.experiment.log({
                "trajectory_completed": traj_idx + 1,
                "total_epochs_trained": clf_epochs_trained
            })
        
        # Test final models
        print("\n=== Testing final dynamics model ===")
        dynamics_test_results = dynamics_trainer.test(dynamics_model, dynamics_data_module)
        
        print("\n=== Testing final CLF model ===")
        clf_test_results = clf_trainer.test(clf_model, clf_data_module)
        
        # Evaluate control policy if requested
        if config.experiment.get("evaluate_control", False):
            print("\n=== Evaluating control policy ===")
            eval_results = evaluate_control_policy(config, clf_model, dynamics_model)
            results["control_evaluation"] = eval_results
        
        # Store results
        results["dynamics"] = {
            "test_results": dynamics_test_results
        }
        results["clf"] = {
            "test_results": clf_test_results
        }
        
        # Log best model paths
        for callback in dynamics_callbacks:
            if isinstance(callback, ModelCheckpoint):
                best_dynamics_path = callback.best_model_path
                print(f"Best dynamics model path: {best_dynamics_path}")
                dynamics_logger.experiment.summary["best_model_path"] = best_dynamics_path
                results["dynamics"]["best_model_path"] = best_dynamics_path
        
        for callback in clf_callbacks:
            if isinstance(callback, ModelCheckpoint):
                best_clf_path = callback.best_model_path
                print(f"Best CLF model path: {best_clf_path}")
                clf_logger.experiment.summary["best_model_path"] = best_clf_path
                results["clf"]["best_model_path"] = best_clf_path
        
        # Clean up environment
        if env is not None:
            env.close()
    
    else:
        raise ValueError(f"Unknown task: {config.experiment.task}")
    
    return results


if __name__ == "__main__":
    main()