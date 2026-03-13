"""
Module for dynamics models (control-affine networks and ensembles).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ControlAffineNetwork(nn.Module):
    """
    Neural network model for learning control-affine dynamics of the form:
    x_dot = f(x) + g(x)u
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
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.dt = dt

        # Needed so _log_predictions_wandb doesn't raise AttributeError (C3)
        self.trajectory_counter = 0

        # Tracking state for the custom training loop
        self.current_epoch: int = 0
        self.global_step: int = 0
        self._wandb_run = None  # set by Trainer before fit()

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

    def _log_metric(self, key: str, val, **kwargs) -> None:
        """Log a scalar metric to W&B if a run is available."""
        if self._wandb_run is not None:
            v = val.item() if torch.is_tensor(val) else float(val)
            self._wandb_run.log({key: v}, step=self.global_step)

    def forward(self, state: torch.Tensor, enable_dropout: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute f(x) and g(x).

        Args:
            state: [batch_size, state_dim]
            enable_dropout: Whether to enable dropout during inference for MC dropout

        Returns:
            Tuple: f(x) [batch_size, state_dim], g(x) [batch_size, state_dim, action_dim]
        """
        batch_size = state.shape[0]

        training = self.training

        if enable_dropout and not training:
            self.train()

        features = self.shared_layers(state)
        f_x = self.f_layer(features)
        g_x_flat = self.g_layer(features)
        g_x = g_x_flat.view(batch_size, self.state_dim, self.action_dim)

        if enable_dropout and not training:
            self.eval()

        return f_x, g_x

    def predict_state_derivative(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        enable_dropout: bool = False
    ) -> torch.Tensor:
        f_x, g_x = self.forward(state, enable_dropout=enable_dropout)
        g_x_u = torch.bmm(g_x, action.unsqueeze(-1)).squeeze(-1)
        return f_x + g_x_u

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: Optional[float] = None,
        enable_dropout: bool = False
    ) -> torch.Tensor:
        if dt is None:
            dt = self.dt
        state_derivative = self.predict_state_derivative(state, action, enable_dropout=enable_dropout)
        return state + dt * state_derivative

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        predicted_next_states = self.predict_next_state(states, actions, dt)
        return F.mse_loss(predicted_next_states, next_states)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]

        loss = self.compute_loss(states, actions, next_states)
        self._log_metric("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        states = batch["states"]

        if "actions" in batch and "next_states" in batch:
            actions = batch["actions"]
            next_states = batch["next_states"]

            loss = self.compute_loss(states, actions, next_states)
            self._log_metric("val_loss", loss)
            self._log_metric("val_dynamics_loss", loss)

            if batch_idx == 0 and self._wandb_run is not None:
                self._log_predictions_wandb(states[:10], actions[:10], next_states[:10])
        else:
            loss = torch.tensor(0.0, device=states.device)
            self._log_metric("val_loss", loss)

        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]

        loss = self.compute_loss(states, actions, next_states)
        self._log_metric("test_loss", loss)
        return {"test_loss": loss}

    def _log_predictions_wandb(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> None:
        with torch.no_grad():
            mean_predictions = self.predict_next_state(states, actions)
            mse_error = F.mse_loss(mean_predictions, next_states).item()

            if self._wandb_run is not None:
                self._wandb_run.log({
                    "ensemble_mse_error": mse_error,
                    "trajectory": self.trajectory_counter,
                    "epoch": self.current_epoch,
                }, step=self.global_step)


class DynamicsEnsemble(nn.Module):
    """
    Ensemble of ControlAffineNetwork models for dynamics learning with uncertainty estimation.
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
        epochs_per_trajectory: int = 5,
        static_norm_k: float = 2.0,
        dynamic_norm_c0: float = 0.05,
        dynamic_norm_alpha: float = 0.01,
        variance_buffer_size: int = 1000
    ) -> None:
        super().__init__()

        self.ensemble_size = ensemble_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        self.mc_dropout_samples = mc_dropout_samples
        self.mc_dropout_enabled = mc_dropout_enabled
        self.learning_rate = learning_rate
        self.dt = dt
        self._epochs_per_trajectory = epochs_per_trajectory
        self.static_norm_k = static_norm_k
        self.dynamic_norm_alpha = dynamic_norm_alpha
        self.variance_buffer_size = variance_buffer_size

        # Register as buffer so it survives checkpoint/resume (M1)
        self.register_buffer(
            "dynamic_norm_c", torch.tensor(dynamic_norm_c0, dtype=torch.float32)
        )

        # Buffer to store variance history for median calculation
        self.variance_history: List[float] = []

        # Trajectory counter to know when to update variance norms
        self.trajectory_counter = 0

        # Tracking state for the custom training loop
        self.current_epoch: int = 0
        self.global_step: int = 0
        self._wandb_run = None  # set by Trainer before fit()

        # Store hparams needed to reconstruct model from checkpoint
        self._hparams = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "ensemble_size": ensemble_size,
            "dropout_rate": dropout_rate,
            "mc_dropout_samples": mc_dropout_samples,
            "mc_dropout_enabled": mc_dropout_enabled,
            "learning_rate": learning_rate,
            "dt": dt,
            "epochs_per_trajectory": epochs_per_trajectory,
            "static_norm_k": static_norm_k,
            "dynamic_norm_c0": dynamic_norm_c0,
            "dynamic_norm_alpha": dynamic_norm_alpha,
            "variance_buffer_size": variance_buffer_size,
        }

        self.models = nn.ModuleList([
            ControlAffineNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                dt=dt
            )
            for _ in range(ensemble_size)
        ])

    def _log_metric(self, key: str, val, **kwargs) -> None:
        """Log a scalar metric to W&B if a run is available."""
        if self._wandb_run is not None:
            v = val.item() if torch.is_tensor(val) else float(val)
            self._wandb_run.log({key: v}, step=self.global_step)

    def forward(
        self,
        state: torch.Tensor,
        enable_dropout: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = [model(state, enable_dropout=enable_dropout) for model in self.models]
        f_outputs = [o[0] for o in outputs]
        g_outputs = [o[1] for o in outputs]
        return torch.mean(torch.stack(f_outputs), dim=0), torch.mean(torch.stack(g_outputs), dim=0)

    def predict_state_derivative(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_individual: bool = False,
        enable_dropout: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        predictions = [
            model.predict_state_derivative(state, action, enable_dropout=enable_dropout)
            for model in self.models
        ]
        if return_individual:
            return predictions
        # Return mean prediction across ensembles
        return torch.mean(torch.stack(predictions), dim=0)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: Optional[float] = None,
        return_individual: bool = False,
        enable_dropout: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if dt is None:
            dt = self.dt
        predictions = [
            model.predict_next_state(state, action, dt, enable_dropout=enable_dropout)
            for model in self.models
        ]
        if return_individual:
            return predictions
        return torch.mean(torch.stack(predictions), dim=0)

    def compute_uncertainty(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_mc_dropout: bool = True
    ) -> torch.Tensor:
        """
        Compute epistemic uncertainty as variance of predictions across ensemble members, and optionally using MC dropout.
        """
        if not use_mc_dropout:
            predictions = self.predict_state_derivative(state, action, return_individual=True)
            if not isinstance(predictions, list):
                predictions = [predictions]
            stacked = torch.stack(predictions)
            return torch.var(stacked, dim=0)
        else:
            all_predictions = []
            for model in self.models:
                for _ in range(self.mc_dropout_samples):
                    with torch.no_grad():
                        pred = model.predict_state_derivative(state, action, enable_dropout=True)
                    all_predictions.append(pred)
            stacked = torch.stack(all_predictions)
            return torch.var(stacked, dim=0)

    def normalize_variance_static(self, variance: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-self.static_norm_k * variance)

    def normalize_variance_dynamic(self, variance: torch.Tensor) -> torch.Tensor:
        return variance / (variance + self.dynamic_norm_c)

    def update_dynamic_normalization_parameter(self, variance: torch.Tensor) -> float:
        flat_variance = variance.flatten().cpu().tolist()
        self.variance_history.extend(flat_variance)
        if len(self.variance_history) > self.variance_buffer_size:
            self.variance_history = self.variance_history[-self.variance_buffer_size:]
        var_median = float(torch.tensor(self.variance_history).median().item())
        # dynamic_norm_c is a registered buffer (M1)
        self.dynamic_norm_c.fill_(
            (1 - self.dynamic_norm_alpha) * self.dynamic_norm_c.item()
            + self.dynamic_norm_alpha * var_median
        )
        return var_median

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        # R8: use torch.stack().mean() across ensemble members
        return torch.stack([
            model.compute_loss(states, actions, next_states, dt)
            for model in self.models
        ]).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        states = batch["states"]

        if "actions" in batch and "next_states" in batch:
            actions = batch["actions"]
            next_states = batch["next_states"]
            loss = self.compute_loss(states, actions, next_states)
            self._log_metric("train_loss", loss)
            self._log_metric("train_dynamics_loss", loss)

            # Ensemble disagreement: std of next-state predictions across members
            with torch.no_grad():
                individual = self.predict_next_state(states, actions, return_individual=True)
                stacked = torch.stack(individual)  # [ensemble_size, batch, state_dim]
                disagreement = stacked.std(dim=0).mean()
            self._log_metric("train_ensemble_disagreement", disagreement)
        else:
            loss = torch.tensor(0.0, device=states.device, requires_grad=True)
            self._log_metric("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        states = batch["states"]

        if "actions" in batch and "next_states" in batch:
            actions = batch["actions"]
            next_states = batch["next_states"]

            loss = self.compute_loss(states, actions, next_states)
            self._log_metric("val_loss", loss)
            self._log_metric("val_dynamics_loss", loss)

            if batch_idx == 0 and self._wandb_run is not None:
                self._log_predictions_wandb(states[:10], actions[:10], next_states[:10])
                self._log_uncertainty_wandb(states[:10], actions[:10])
        else:
            loss = torch.tensor(0.0, device=states.device)
            self._log_metric("val_loss", loss)

        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]

        with torch.no_grad():
            pred_next_states = self.predict_next_state(
                states, actions, enable_dropout=self.mc_dropout_enabled
            )
            loss = F.mse_loss(pred_next_states, next_states)

        self._log_metric("test_loss", loss)
        return {"test_loss": loss}

    def on_train_epoch_start(self) -> None:
        """
        Hook called when a training epoch begins.
        Used to track trajectory boundaries and update variance normalization parameters.
        """
        epochs_per_trajectory = self._epochs_per_trajectory

        if self.current_epoch == 0:
            self.trajectory_counter = 0
            self.variance_history = []

            if self._wandb_run is not None:
                self._wandb_run.log({
                    "trajectory": self.trajectory_counter,
                    "dynamic_norm_parameter_c": self.dynamic_norm_c.item(),
                }, step=self.global_step)

            print(f"Starting trajectory {self.trajectory_counter + 1} at epoch {self.current_epoch}")

        if self.current_epoch > 0 and self.current_epoch % epochs_per_trajectory == 0:
            self.trajectory_counter += 1
            self.variance_history = []

            if self._wandb_run is not None:
                self._wandb_run.log({
                    "trajectory": self.trajectory_counter,
                    "dynamic_norm_parameter_c": self.dynamic_norm_c.item(),
                    "new_trajectory_marker": 1.0,
                }, step=self.global_step)

            print(f"Starting trajectory {self.trajectory_counter + 1} at epoch {self.current_epoch}")

    def on_train_epoch_end(self) -> None:
        """Log dynamic normalization parameter at the end of each training epoch."""
        self._log_metric("dynamic_norm_parameter_c", self.dynamic_norm_c)

    def _log_predictions_wandb(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> None:
        with torch.no_grad():
            mean_predictions = self.predict_next_state(states, actions)
            mse_error = F.mse_loss(mean_predictions, next_states).item()

            if self._wandb_run is not None:
                self._wandb_run.log({
                    "ensemble_mse_error": mse_error,
                    "trajectory": self.trajectory_counter,
                    "epoch": self.current_epoch,
                }, step=self.global_step)

    def _log_uncertainty_wandb(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> None:
        with torch.no_grad():
            raw_variance = self.compute_uncertainty(states, actions, use_mc_dropout=False)
            static_norm_variance = self.normalize_variance_static(raw_variance)
            dynamic_norm_variance = self.normalize_variance_dynamic(raw_variance)

            mean_raw_variance = raw_variance.mean().item()
            mean_static_norm = static_norm_variance.mean().item()
            mean_dynamic_norm = dynamic_norm_variance.mean().item()

            # Update dynamic normalization parameter; use returned median (W2)
            var_median = self.update_dynamic_normalization_parameter(raw_variance)

            if self._wandb_run is not None:
                self._wandb_run.log({
                    "raw_variance_mean": mean_raw_variance,
                    "static_norm_variance_mean": mean_static_norm,
                    "dynamic_norm_variance_mean": mean_dynamic_norm,
                    "variance_median": var_median,
                    "dynamic_norm_parameter_c": self.dynamic_norm_c.item(),
                    "trajectory": self.trajectory_counter,
                    "epoch": self.current_epoch,
                }, step=self.global_step)

    def save_checkpoint(self, path: str, val_loss: Optional[float] = None) -> None:
        """Save model to a checkpoint file."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "hparams": self._hparams,
            "val_loss": val_loss,
            "epoch": self.current_epoch,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = "cpu") -> "DynamicsEnsemble":
        """Load model from a checkpoint file saved by save_checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)
        hparams = checkpoint["hparams"]
        model = cls(**hparams)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# Backward-compatibility aliases (old Lightning names still importable)
ControlAffineNetworkLightning = ControlAffineNetwork
DynamicsEnsembleLightning = DynamicsEnsemble
