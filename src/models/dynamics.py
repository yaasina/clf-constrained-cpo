"""
Module for PyTorch Lightning implementations of dynamics models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional, Union
import pytorch_lightning as pl
import numpy as np
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ControlAffineNetworkLightning(pl.LightningModule):
    """
    Neural network model for learning control-affine dynamics of the form:
    x_dot = f(x) + g(x)u
    
    Implemented with PyTorch Lightning for better training management.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        dt: float = 0.05
    ) -> None:
        """
        Initialize the control-affine network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout probability for regularization
            learning_rate: Learning rate for optimizer
            dt: Time step for Euler integration
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.dt = dt
        
        # Shared layers with dropout
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        # f(x) - autonomous dynamics
        self.f_layer = nn.Linear(hidden_dim, state_dim)
        
        # g(x) - control matrix
        self.g_layer = nn.Linear(hidden_dim, state_dim * action_dim)
        
    def forward(self, state: torch.Tensor, enable_dropout: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute f(x) and g(x).
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            enable_dropout: Whether to enable dropout during inference for MC dropout
            
        Returns:
            Tuple containing:
                f(x): Autonomous dynamics [batch_size, state_dim]
                g(x): Control matrix [batch_size, state_dim, action_dim]
        """
        batch_size = state.shape[0]
        
        # Save current training state
        training = self.training
        
        # Enable dropout during inference if MC dropout is enabled
        if enable_dropout and not training:
            self.train()
        
        # Compute shared features
        features = self.shared_layers(state)
        
        # Compute f(x)
        f_x = self.f_layer(features)
        
        # Compute g(x) and reshape to [batch_size, state_dim, action_dim]
        g_x_flat = self.g_layer(features)
        g_x = g_x_flat.view(batch_size, self.state_dim, self.action_dim)
        
        # Restore original training state
        if enable_dropout and not training:
            self.eval()
        
        return f_x, g_x
    
    def predict_state_derivative(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        enable_dropout: bool = False
    ) -> torch.Tensor:
        """
        Predict the state derivative using the control-affine model.
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            action: Batch of action vectors [batch_size, action_dim]
            enable_dropout: Whether to enable dropout during inference for MC dropout
            
        Returns:
            Predicted state derivative [batch_size, state_dim]
        """
        f_x, g_x = self.forward(state, enable_dropout=enable_dropout)
        
        # Compute x_dot = f(x) + g(x)u
        # g_x shape: [batch_size, state_dim, action_dim]
        # action shape: [batch_size, action_dim]
        # Need to do batch matrix-vector product
        g_x_u = torch.bmm(g_x, action.unsqueeze(-1)).squeeze(-1)
        
        x_dot = f_x + g_x_u
        return x_dot
    
    def predict_next_state(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        dt: Optional[float] = None,
        enable_dropout: bool = False
    ) -> torch.Tensor:
        """
        Predict the next state using Euler integration.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action [batch_size, action_dim]
            dt: Time step for Euler integration (uses self.dt if None)
            enable_dropout: Whether to enable dropout during inference for MC dropout
            
        Returns:
            Predicted next state [batch_size, state_dim]
        """
        if dt is None:
            dt = self.dt
            
        state_derivative = self.predict_state_derivative(state, action, enable_dropout=enable_dropout)
        next_state = state + dt * state_derivative
        return next_state
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and actual next states.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            next_states: Batch of next states [batch_size, state_dim]
            dt: Time step used for Euler integration (uses self.dt if None)
            
        Returns:
            MSE loss value
        """
        predicted_next_states = self.predict_next_state(states, actions, dt)
        loss = F.mse_loss(predicted_next_states, next_states)
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states, actions, next_states
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        
        # Compute loss
        loss = self.compute_loss(states, actions, next_states)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states, actions, next_states
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        
        # Compute loss
        loss = self.compute_loss(states, actions, next_states)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # If it's the first validation batch of an epoch, visualize predictions
        if batch_idx == 0 and self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger):
            self._log_predictions_wandb(states[:10], actions[:10], next_states[:10])
            
        return {"val_loss": loss}
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states, actions, next_states
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        
        # Compute loss (dropout is disabled by default during testing)
        loss = self.compute_loss(states, actions, next_states)
        
        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        
        return {"test_loss": loss}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for PyTorch Lightning.
        
        Returns:
            Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _log_predictions_wandb(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        next_states: torch.Tensor
    ) -> None:
        """
        Log streamlined model predictions to Weights & Biases, focusing only on
        essential learning curve visualizations.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            next_states: Ground truth next states [batch_size, state_dim]
        """
        with torch.no_grad():
            # Get mean predictions
            mean_predictions = self.predict_next_state(states, actions)
            
            # Compute MSE error
            mse_error = F.mse_loss(mean_predictions, next_states).item()
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                # Log only the essential learning curve metrics
                metrics = {
                    "ensemble_mse_error": mse_error,
                    "trajectory": self.trajectory_counter,
                    "epoch": self.current_epoch
                }
                wandb.log(metrics)


class DynamicsEnsembleLightning(pl.LightningModule):
    """
    Ensemble of control-affine networks for dynamics learning with uncertainty estimation.
    Implemented with PyTorch Lightning for better training management.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64,
        ensemble_size: int = 5,
        dropout_rate: float = 0.1,
        mc_dropout_samples: int = 20,
        mc_dropout_enabled: bool = False,
        learning_rate: float = 1e-3,
        dt: float = 0.05,
        # Variance normalization parameters
        static_norm_k: float = 2.0,
        dynamic_norm_c0: float = 0.05,
        dynamic_norm_alpha: float = 0.01,
        variance_buffer_size: int = 1000
    ) -> None:
        """
        Initialize the dynamics ensemble.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            ensemble_size: Number of models in the ensemble
            dropout_rate: Dropout probability for regularization
            mc_dropout_samples: Number of samples for MC dropout uncertainty estimation
            mc_dropout_enabled: Whether to use MC dropout during inference (default: False)
            learning_rate: Learning rate for optimizer
            dt: Time step for Euler integration
            static_norm_k: Scaling parameter for static normalization: 1-exp(-k*var)
            dynamic_norm_c0: Initial value for dynamic normalization parameter
            dynamic_norm_alpha: Learning rate for dynamic normalization parameter update
            variance_buffer_size: Size of the buffer for variance history (for computing median)
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.ensemble_size = ensemble_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        self.mc_dropout_samples = mc_dropout_samples
        self.mc_dropout_enabled = mc_dropout_enabled
        self.learning_rate = learning_rate
        self.dt = dt
        
        # Variance normalization parameters
        self.static_norm_k = static_norm_k
        self.dynamic_norm_c = dynamic_norm_c0  # Current value of c_t
        self.dynamic_norm_c0 = dynamic_norm_c0  # Initial value for c_t
        self.dynamic_norm_alpha = dynamic_norm_alpha
        self.variance_buffer_size = variance_buffer_size
        
        # Buffer to store variance history for median calculation
        self.variance_history = []
        
        # Trajectory counter to know when to update variance norms
        self.trajectory_counter = 0
        
        # Create ensemble of models with dropout
        self.models = nn.ModuleList([
            ControlAffineNetworkLightning(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                dt=dt
            )
            for _ in range(ensemble_size)
        ])
        
    def forward(
        self, 
        state: torch.Tensor,
        enable_dropout: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all models in the ensemble.
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            enable_dropout: Whether to enable dropout during inference for MC dropout
            
        Returns:
            Tuple containing:
                f(x): Mean autonomous dynamics [batch_size, state_dim]
                g(x): Mean control matrix [batch_size, state_dim, action_dim]
        """
        # Get predictions from all models
        outputs = [model(state, enable_dropout=enable_dropout) for model in self.models]
        
        # Separate f(x) and g(x) components
        f_outputs = [output[0] for output in outputs]
        g_outputs = [output[1] for output in outputs]
        
        # Return mean predictions
        mean_f = torch.mean(torch.stack(f_outputs), dim=0)
        mean_g = torch.mean(torch.stack(g_outputs), dim=0)
        
        return mean_f, mean_g
    
    def predict_state_derivative(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        return_individual: bool = False,
        enable_dropout: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Predict state derivatives from all models.
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            action: Batch of action vectors [batch_size, action_dim]
            return_individual: If True, return predictions from each model
            enable_dropout: Whether to enable dropout during inference for MC dropout
            
        Returns:
            If return_individual is True:
                List of predictions from each model [ensemble_size, batch_size, state_dim]
            Else:
                Mean prediction across ensemble [batch_size, state_dim]
        """
        predictions = [
            model.predict_state_derivative(state, action, enable_dropout=enable_dropout) 
            for model in self.models
        ]
        
        if return_individual:
            return predictions
        
        # Return mean prediction
        return torch.mean(torch.stack(predictions), dim=0)
    
    def predict_next_state(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        dt: Optional[float] = None,
        return_individual: bool = False,
        enable_dropout: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Predict next states from all models.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action [batch_size, action_dim]
            dt: Time step for Euler integration (uses self.dt if None)
            return_individual: If True, return predictions from each model
            enable_dropout: Whether to enable dropout during inference for MC dropout
            
        Returns:
            If return_individual is True:
                List of next state predictions [ensemble_size, batch_size, state_dim]
            Else:
                Mean next state prediction [batch_size, state_dim]
        """
        if dt is None:
            dt = self.dt
            
        predictions = [
            model.predict_next_state(state, action, dt, enable_dropout=enable_dropout) 
            for model in self.models
        ]
        
        if return_individual:
            return predictions
        
        # Return mean prediction
        return torch.mean(torch.stack(predictions), dim=0)
    
    def compute_uncertainty(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        use_mc_dropout: bool = True
    ) -> torch.Tensor:
        """
        Compute epistemic uncertainty as the variance of predictions across the ensemble.
        Optionally uses Monte Carlo dropout for additional uncertainty estimation.
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            action: Batch of action vectors [batch_size, action_dim]
            use_mc_dropout: If True, use MC dropout for uncertainty estimation
            
        Returns:
            Uncertainty measure [batch_size, state_dim]
        """
        if not use_mc_dropout:
            # Original ensemble uncertainty
            predictions = self.predict_state_derivative(state, action, return_individual=True)
            # Ensure predictions is treated as a list of Tensors
            if not isinstance(predictions, list):
                predictions = [predictions]
            stacked_predictions = torch.stack(predictions)  # [ensemble_size, batch_size, state_dim]
            
            # Compute variance across the ensemble dimension
            uncertainty = torch.var(stacked_predictions, dim=0)  # [batch_size, state_dim]
        else:
            # Monte Carlo dropout uncertainty
            all_predictions = []
            
            # For each model in the ensemble
            for model in self.models:
                # Get multiple predictions with different dropout masks
                for _ in range(self.mc_dropout_samples):
                    with torch.no_grad():
                        prediction = model.predict_state_derivative(state, action, enable_dropout=True)
                    all_predictions.append(prediction)
            
            # Stack all MC predictions [ensemble_size * mc_samples, batch_size, state_dim]
            stacked_predictions = torch.stack(all_predictions)
            
            # Compute variance across all MC samples
            uncertainty = torch.var(stacked_predictions, dim=0)  # [batch_size, state_dim]
        
        return uncertainty
    
    def normalize_variance_static(self, variance: torch.Tensor) -> torch.Tensor:
        """
        Apply static normalization to the variance: 1-exp(-k*var)
        Maps variance from [0, inf] to [0, 1] with k as scaling parameter.
        
        Args:
            variance: Raw variance tensor
            
        Returns:
            Normalized variance within [0, 1]
        """
        return 1.0 - torch.exp(-self.static_norm_k * variance)
    
    def normalize_variance_dynamic(self, variance: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic normalization to the variance: var/(var+c_t)
        Maps variance from [0, inf] to [0, 1] using the dynamic parameter c_t.
        
        Args:
            variance: Raw variance tensor
            
        Returns:
            Normalized variance within [0, 1]
        """
        return variance / (variance + self.dynamic_norm_c)
    
    def update_dynamic_normalization_parameter(self, variance: torch.Tensor) -> None:
        """
        Update the dynamic normalization parameter c_t based on the median of variance values.
        Formula: c_t+1 = (1-alpha) * c_t + alpha * var_median(t+1)
        
        Args:
            variance: Current batch of variance values
        """
        # Flatten the variance tensor and convert to list
        flat_variance = variance.flatten().cpu().tolist()
        
        # Add to the variance history buffer
        self.variance_history.extend(flat_variance)
        
        # Maintain the buffer size limit
        if len(self.variance_history) > self.variance_buffer_size:
            # Keep only the most recent values
            self.variance_history = self.variance_history[-self.variance_buffer_size:]
        
        # Calculate median of the variance history
        var_median = torch.tensor(self.variance_history).median().item()
        
        # Update c_t using the exponential moving average formula
        self.dynamic_norm_c = (1 - self.dynamic_norm_alpha) * self.dynamic_norm_c + \
                              self.dynamic_norm_alpha * var_median
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and actual next states for each model in the ensemble.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            next_states: Batch of next states [batch_size, state_dim]
            dt: Time step used for Euler integration (uses self.dt if None)
            
        Returns:
            Average MSE loss value across all models
        """
        total_loss = 0.0
        
        for model in self.models:
            model_loss = model.compute_loss(states, actions, next_states, dt)
            total_loss += model_loss
        
        # Average loss across all models
        return total_loss / self.ensemble_size
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states, actions, next_states
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        
        # Compute loss
        loss = self.compute_loss(states, actions, next_states)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": loss}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states, actions, next_states
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        
        # Compute loss
        loss = self.compute_loss(states, actions, next_states)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # If it's the first validation batch of an epoch, visualize predictions and uncertainty
        if batch_idx == 0 and self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger):
            self._log_predictions_wandb(states[:10], actions[:10], next_states[:10])
            self._log_uncertainty_wandb(states[:10], actions[:10])
            
        return {"val_loss": loss}
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states, actions, next_states
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        
        # Use mc_dropout_enabled flag to determine if dropout should be active during inference
        # By default, dropout is disabled during inference (evaluation mode)
        with torch.no_grad():
            pred_next_states = self.predict_next_state(
                states, 
                actions, 
                enable_dropout=self.mc_dropout_enabled
            )
            
            # Compute MSE loss between the predicted and actual next states
            loss = F.mse_loss(pred_next_states, next_states)
        
        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        
        return {"test_loss": loss}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for PyTorch Lightning.
        
        Returns:
            Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_train_epoch_start(self) -> None:
        """
        Hook that is called when a training epoch begins.
        Used to track when a new trajectory is collected and update variance normalization parameters.
        """
        super().on_train_epoch_start()
        
        # Get configuration parameters for trajectory handling
        epochs_per_trajectory = self.trainer.datamodule.hparams.get('epochs_per_trajectory', 5) if hasattr(self.trainer, 'datamodule') else 5
        
        # Initialize on first epoch
        if self.current_epoch == 0:
            # Reset trajectory counter and variance history
            self.trajectory_counter = 0
            self.variance_history = []
            
            # Log initial values to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({
                    "trajectory": self.trajectory_counter,
                    "dynamic_norm_parameter_c": self.dynamic_norm_c
                })
            
            print(f"Starting trajectory {self.trajectory_counter+1} at epoch {self.current_epoch}")
        
        # Check if we're starting a new trajectory
        # We determine this by checking if the current epoch is at the start of a trajectory block
        if self.current_epoch > 0 and self.current_epoch % epochs_per_trajectory == 0:
            self.trajectory_counter += 1
            
            # Reset variance history for the new trajectory (to ensure it adapts to the new data)
            self.variance_history = []
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({
                    "trajectory": self.trajectory_counter,
                    "dynamic_norm_parameter_c": self.dynamic_norm_c,
                    "new_trajectory_marker": 1.0  # Add marker to easily identify new trajectories in plots
                })
            
            print(f"Starting trajectory {self.trajectory_counter+1} at epoch {self.current_epoch}")
            
            # Note: We don't reset dynamic_norm_c to its initial value
            # Instead we let it adapt continuously across trajectories
            # This allows the normalization to evolve throughout training
    
    def _log_predictions_wandb(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        next_states: torch.Tensor
    ) -> None:
        """
        Log streamlined model predictions to Weights & Biases, focusing only on
        essential learning curve visualizations.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            next_states: Ground truth next states [batch_size, state_dim]
        """
        with torch.no_grad():
            # Get mean predictions
            mean_predictions = self.predict_next_state(states, actions)
            
            # Compute MSE error
            mse_error = F.mse_loss(mean_predictions, next_states).item()
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                # Log only the essential learning curve metrics
                metrics = {
                    "ensemble_mse_error": mse_error,
                    "trajectory": self.trajectory_counter,
                    "epoch": self.current_epoch
                }
                wandb.log(metrics)
    
    def _log_uncertainty_wandb(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> None:
        """
        Log streamlined uncertainty visualization to Weights & Biases.
        Only logs essential metrics: raw variance, normalized variances (static and dynamic),
        variance median, and dynamic normalization parameter evolution over trajectories.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
        """
        with torch.no_grad():
            # Compute ensemble uncertainty (variance) without MC dropout
            raw_variance = self.compute_uncertainty(states, actions, use_mc_dropout=False)
            
            # Normalize variance using both methods
            static_norm_variance = self.normalize_variance_static(raw_variance)
            dynamic_norm_variance = self.normalize_variance_dynamic(raw_variance)
            
            # Calculate mean variance across batch and dimensions
            # We only care about the overall mean, not per-dimension values
            mean_raw_variance = raw_variance.mean().item()
            mean_static_norm = static_norm_variance.mean().item()
            mean_dynamic_norm = dynamic_norm_variance.mean().item()
            
            # Update dynamic normalization parameter based on current variance
            # and calculate the current median value
            flat_variance = raw_variance.flatten().cpu().tolist()
            
            # Add to the variance history buffer
            self.variance_history.extend(flat_variance)
            
            # Maintain the buffer size limit
            if len(self.variance_history) > self.variance_buffer_size:
                # Keep only the most recent values
                self.variance_history = self.variance_history[-self.variance_buffer_size:]
            
            # Calculate median of the variance history
            var_median = torch.tensor(self.variance_history).median().item()
            
            # Update c_t using the exponential moving average formula
            self.dynamic_norm_c = (1 - self.dynamic_norm_alpha) * self.dynamic_norm_c + \
                                  self.dynamic_norm_alpha * var_median
            
            # Log essential metrics to wandb with trajectory information
            if isinstance(self.logger, pl.loggers.WandbLogger):
                # Log the current variance values for this epoch
                metrics = {
                    "raw_variance_mean": mean_raw_variance,
                    "static_norm_variance_mean": mean_static_norm, 
                    "dynamic_norm_variance_mean": mean_dynamic_norm,
                    "variance_median": var_median,
                    "dynamic_norm_parameter_c": self.dynamic_norm_c,
                    "trajectory": self.trajectory_counter,
                    "epoch": self.current_epoch
                }
                
                # Calculate trajectory-level metrics (average of all epochs in a trajectory)
                # This is a running calculation that will be finalized at the end of the trajectory
                wandb.log(metrics)
                
                # Skip creating the bar charts and other visualizations that aren't needed