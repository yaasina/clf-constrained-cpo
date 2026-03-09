"""
Module for PyTorch Lightning implementations of Control Lyapunov Function models.
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


class CLFNetworkLightning(pl.LightningModule):
    """
    Neural network model for learning a Control Lyapunov Function (CLF).
    The CLF is of the form L(x) = 0.5 * (NN(x))^2, where NN(x) is the neural network output.
    This formulation ensures that L(x) is positive definite and L(0) = 0.
    
    Implemented with PyTorch Lightning for better training management.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        nonlinearity: nn.Module = nn.Tanh(),
        output_nonlinearity: Optional[nn.Module] = None,
        loss: Optional[Dict[str, float]] = None,
        equilibrium: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initialize the CLF network.

        Args:
            state_dim: Dimension of the state space
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout probability for regularization
            learning_rate: Learning rate for optimizer
            nonlinearity: Activation function for hidden layers
            output_nonlinearity: Activation function for output layer (None for linear output)
            loss: Dictionary of loss hyperparameters (alpha1, alpha2, alpha3, alpha4)
            equilibrium: Equilibrium state tensor [state_dim]. Defaults to the zero vector.
                         Registered as a buffer so it moves with the model and is saved
                         in checkpoints.
        """
        super().__init__()

        # Exclude nn.Module args and tensor from hyperparameter serialization (C4 / L6)
        self.save_hyperparameters(ignore=["nonlinearity", "output_nonlinearity", "equilibrium"])

        self.state_dim = state_dim
        self.output_nonlinearity = output_nonlinearity
        self.learning_rate = learning_rate

        # Set loss hyperparameters with defaults
        self.loss_params = loss or {}
        self.alpha1 = self.loss_params.get('alpha1', 1.0)
        self.alpha2 = self.loss_params.get('alpha2', 0.1)
        self.alpha3 = self.loss_params.get('alpha3', 1.0)
        self.alpha4 = self.loss_params.get('alpha4', 1.0)

        # Register equilibrium as a buffer so it moves to the right device automatically
        # and is saved / restored with the checkpoint (L6)
        _eq = equilibrium.clone().detach().float() if equilibrium is not None else torch.zeros(state_dim)
        self.register_buffer("equilibrium", _eq)

        # Define neural network architecture with dropout
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nonlinearity,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)  # Output a scalar
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute NN(x).
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            
        Returns:
            Network output [batch_size, 1]
        """
        # For equilibrium point (origin), ensure output is exactly zero
        if torch.all(torch.abs(state) < 1e-6):
            # Keep the computational graph alive through state so gradients can flow
            return torch.zeros(state.shape[0], 1, device=state.device) + state.sum() * 0.0
        
        # Normal forward pass through the network layers
        output = self.layers(state)
        
        # Apply output nonlinearity if specified
        if self.output_nonlinearity is not None:
            output = self.output_nonlinearity(output)
            
        return output
    
    def compute_clf(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the Control Lyapunov Function value L(x) = 0.5 * (NN(x))^2.

        Args:
            state: Batch of state vectors [batch_size, state_dim]

        Returns:
            CLF values [batch_size, 1]
        """
        nn_output = self(state)

        # Calculate V(x) = 0.5 * (NN(x))^2
        clf_value = 0.5 * torch.pow(nn_output, 2)

        return clf_value
    
    def gradient(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the CLF with respect to the state.

        Uses a single vectorised autograd call over the entire batch:
        d(ΣᵢV(xᵢ))/d(xⱼ) = dV(xⱼ)/d(xⱼ) because the network has no
        cross-sample interactions.  This replaces an O(N) per-sample loop
        (G2) and eliminates the wasted full-batch forward pass (G3) and
        the in-place assignment to a zeros tensor (G1).

        Args:
            state: Batch of state vectors [batch_size, state_dim]

        Returns:
            Gradient of L(x) with respect to x [batch_size, state_dim]
        """
        state_grad = state.detach().clone().requires_grad_(True)
        clf_sum = self.compute_clf(state_grad).sum()
        gradients = torch.autograd.grad(
            clf_sum, state_grad, create_graph=True
        )[0]
        return gradients
    
    def _compute_gradient_fd(self, state: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Compute gradient using finite differences as a fallback.
        
        Args:
            state: Single state vector [state_dim]
            epsilon: Small step size for finite difference
            
        Returns:
            Gradient vector [state_dim]
        """
        # Create a detached copy of the state
        state_np = state.detach().cpu().numpy()
        state_dim = state_np.shape[0]
        gradient = np.zeros(state_dim)
        
        # Compute base CLF value for unperturbed state
        base_state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_value = self.compute_clf(base_state).item()
        
        # Compute finite difference approximation of gradient
        for i in range(state_dim):
            # Create perturbed state (only perturb one dimension at a time)
            perturbed_state_np = state_np.copy()
            perturbed_state_np[i] += epsilon
            
            # Compute CLF value for perturbed state
            perturbed_state = torch.tensor(perturbed_state_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                perturbed_value = self.compute_clf(perturbed_state).item()
            
            # Compute partial derivative
            gradient[i] = (perturbed_value - base_value) / epsilon
        
        return torch.tensor(gradient, dtype=torch.float32, device=state.device)
    
    def lie_derivatives(
        self, 
        state: torch.Tensor, 
        dynamics_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Lie derivatives of the CLF with respect to the dynamics.
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            dynamics_model: Model of the system dynamics (f(x) and g(x))
            
        Returns:
            Tuple containing:
                L_f_V: Lie derivative of V along f [batch_size, 1]
                L_g_V: Lie derivative of V along g [batch_size, action_dim]
        """
        # Compute gradient of CLF
        grad_L = self.gradient(state)
        
        # Get dynamics
        try:
            with torch.no_grad():
                # First try to get dynamics using the forward method
                try:
                    f_x, g_x = dynamics_model(state)
                except Exception as e:
                    # If not an ensemble, try individual model
                    if hasattr(dynamics_model, 'models') and len(dynamics_model.models) > 0:
                        # Try the first model in the ensemble
                        f_x, g_x = dynamics_model.models[0](state)
                    elif hasattr(dynamics_model, 'f') and hasattr(dynamics_model, 'g'):
                        # If model has explicit f and g methods
                        f_x = dynamics_model.f(state)
                        g_x = dynamics_model.g(state)
                    else:
                        raise ValueError(f"Failed to get dynamics: {e}")
        except Exception as e:
            print(f"Error getting dynamics for Lie derivative: {e}")
            raise
        
        # Compute Lie derivatives
        # L_f_V = ∇L · f(x)
        L_f_V = torch.sum(grad_L * f_x, dim=1, keepdim=True)
        
        # L_g_V = ∇L · g(x)
        # Handle different possible shapes of g_x:
        if g_x.dim() == 3:  # Format [batch_size, state_dim, action_dim]
            # Use batch matrix multiplication
            L_g_V = torch.bmm(grad_L.unsqueeze(1), g_x).squeeze(1)
        else:  # Format [batch_size, state_dim * action_dim] or other
            # Try to determine action_dim from the shape
            try:
                action_dim = g_x.shape[1] // state.shape[1]
                g_x_reshaped = g_x.reshape(g_x.shape[0], state.shape[1], action_dim)
                L_g_V = torch.bmm(grad_L.unsqueeze(1), g_x_reshaped).squeeze(1)
            except Exception as e:
                # If reshaping fails, print debug info and raise error
                print(f"Error reshaping g_x. Shapes - state: {state.shape}, g_x: {g_x.shape}, grad_L: {grad_L.shape}")
                raise e
        
        return L_f_V, L_g_V
    
    def compute_lie_derivative_with_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dynamics_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute the total Lie derivative of the CLF with the given action.
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            action: Batch of action vectors [batch_size, action_dim]
            dynamics_model: Model of the system dynamics
            
        Returns:
            Total Lie derivative L_f_V + L_g_V·u [batch_size, 1]
        """
        L_f_V, L_g_V = self.lie_derivatives(state, dynamics_model)
        
        # Compute L_f_V + L_g_V·u
        L_g_V_u = torch.sum(L_g_V * action, dim=1, keepdim=True)
        L_dot = L_f_V + L_g_V_u
        
        return L_dot
    
    def compute_clf_loss(
        self,
        states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute loss for the CLF.
        
        Args:
            states: Batch of state vectors [batch_size, state_dim]
            targets: Optional target CLF values [batch_size, 1]
            epsilon: Small constant for numerical stability
            
        Returns:
            Loss value
        """
        clf_values = self.compute_clf(states)
        
        if targets is not None:
            # Supervised loss if targets are provided
            loss = F.mse_loss(clf_values, targets)
        else:
            # Unsupervised loss: ensure CLF is positive definite
            # L(x) should be positive for all non-zero states
            state_norms = torch.norm(states, dim=1, keepdim=True)
            zero_mask = state_norms < epsilon
            
            # Create target values proportional to state norm
            target_values = 0.5 * state_norms**2
            target_values[zero_mask] = 0.0
            
            # Compute MSE loss
            loss = F.mse_loss(clf_values, target_values)
            
            # Add penalty for negative CLF values (should not happen due to the squaring)
            negative_penalty = torch.relu(-clf_values).mean()
            
            loss = loss + 10.0 * negative_penalty
            
        return loss
    
    def compute_lie_derivative_loss(
        self,
        states: torch.Tensor,
        dynamics_model: nn.Module,
        lambda_lie: float = 1.0,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute loss to ensure the CLF's Lie derivative is negative.
        
        Args:
            states: Batch of state vectors [batch_size, state_dim]
            dynamics_model: Model of the system dynamics
            lambda_lie: Weight for the Lie derivative loss term
            epsilon: Small constant for numerical stability
            
        Returns:
            Loss value
        """
        # Exclude points very close to the equilibrium
        state_norms = torch.norm(states, dim=1, keepdim=True)
        non_zero_mask = state_norms > epsilon
        
        if not torch.any(non_zero_mask):
            # If all points are near equilibrium, return zero loss
            return torch.tensor(0.0, device=states.device, requires_grad=True)
        
        # Compute Lie derivatives
        L_f_V, L_g_V = self.lie_derivatives(states, dynamics_model)
        
        # For true CLF condition, there must exist a u such that L_f_V + L_g_V * u < 0
        # For stabilizing control, choose u = -L_g_V.T if L_g_V is non-zero
        
        # Check where L_g_V has significant magnitude
        L_g_V_norm = torch.norm(L_g_V, dim=1, keepdim=True)
        significant_control = L_g_V_norm > epsilon
        
        # For points with significant control authority, compute u = -L_g_V / (L_g_V_norm + eps)
        # This normalises the control direction without requiring boolean indexing (R2)
        u = -L_g_V / (L_g_V_norm + epsilon)
        
        # Compute Lie derivative with the computed control
        L_dot = L_f_V + torch.sum(L_g_V * u, dim=1, keepdim=True)
        
        # For points with significant control authority, Lie derivative should be negative
        # For points with insignificant control, we want L_f_V to be negative already
        
        # Compute loss: we want L_dot < -alpha * V 
        # where alpha is a small positive constant for exponential convergence
        alpha = 0.1
        clf_values = self.compute_clf(states)
        desired_L_dot = -alpha * clf_values
        
        # Compute loss as a hinge loss: max(0, L_dot - desired_L_dot)
        # This penalizes when L_dot > desired_L_dot
        lie_loss = torch.relu(L_dot - desired_L_dot).mean()
        
        return lambda_lie * lie_loss
    
    def compute_self_supervised_clf_loss(
        self,
        states: torch.Tensor,
        dynamics_model: nn.Module,
        qp_solver,
        next_states: Optional[torch.Tensor] = None,
        dt: float = 0.05,
        alpha1: float = 1.0,  # Weight for equilibrium term
        alpha2: float = 0.1,  # Weight for relaxation variable term
        alpha3: float = 1.0,  # Weight for lie derivative term
        alpha4: float = 1.0,  # Weight for CLF decrease over time term
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the self-supervised CLF loss according to the formula:
        L(θ) = α₁V_θ(0) + (α₂/B)∑ᵢr_i + (α₃/B)∑ᵢmax(L_f V_θ(x_i) + L_g V_θ(x_i)u_i, 0) + (α₄/B)∑ᵢmax((V_θ(x_i⁺) - V_θ(x_i))/Δt, 0)
        
        Args:
            states: Batch of state vectors [batch_size, state_dim]
            dynamics_model: Model of the system dynamics
            qp_solver: QP solver for computing optimal controls and relaxation variables
            next_states: Optional batch of next state vectors [batch_size, state_dim]
            dt: Time step for computing temporal difference
            alpha1-alpha4: Weights for the different loss terms
            
        Returns:
            Dictionary containing individual loss terms and total loss
        """
        batch_size = states.shape[0]
        device = states.device

        # Term 1: V_θ(eq) — CLF value at equilibrium should be 0.
        # Use self.equilibrium buffer; no hard-coded pendulum import (L6).
        equilibrium_value = self.compute_clf(self.equilibrium.to(device).unsqueeze(0))
        loss_equilibrium = alpha1 * equilibrium_value

        # Compute Lie derivatives for the batch once — reused in Term 3 (G5)
        L_f_V, L_g_V = self.lie_derivatives(states, dynamics_model)

        # Solve QP for each state to get optimal controls and relaxation variables
        qp_results = qp_solver.solve_batch(states, self, dynamics_model, batch_size=batch_size)

        u_values = qp_results['u_values']  # [batch_size, action_dim]
        r_values = qp_results['r_values']  # [batch_size] or [batch_size, 1]

        # Term 2: (α₂/B)∑ᵢrᵢ — minimise relaxation variables.
        # Filter inf values produced by failed QP solves to keep loss finite (L7).
        valid_r = r_values[~torch.isinf(r_values)]
        if valid_r.numel() > 0:
            loss_relaxation = alpha2 * valid_r.mean()
        else:
            loss_relaxation = torch.tensor(0.0, device=device, requires_grad=False)

        # Term 3: (α₃/B)∑ᵢmax(L_f V + L_g V·uᵢ, 0) — reuse already-computed L_f_V, L_g_V (G5)
        # u_values needs to be broadcastable with L_g_V [batch_size, action_dim]
        if u_values.dim() == 1:
            u_values = u_values.unsqueeze(-1)
        L_dot = L_f_V + torch.sum(L_g_V * u_values, dim=1, keepdim=True)
        loss_lie_derivative = alpha3 * torch.relu(L_dot).mean()
        
        # Term 4: (α₄/B)∑ᵢmax((V_θ(x_i⁺) - V_θ(x_i))/Δt, 0) - Ensure CLF decreases over time
        loss_temporal = torch.tensor(0.0, device=device)
        if next_states is not None:
            current_values = self.compute_clf(states)
            next_values = self.compute_clf(next_states)
            
            # Compute temporal difference rate (V(x+) - V(x))/dt
            v_dot = (next_values - current_values) / dt
            
            # We want this to be negative, so penalize positive values
            loss_temporal = alpha4 * torch.mean(torch.relu(v_dot))
        
        # Combine all terms
        total_loss = loss_equilibrium + loss_relaxation + loss_lie_derivative + loss_temporal
        
        return {
            "loss": total_loss,
            "loss_equilibrium": loss_equilibrium,
            "loss_relaxation": loss_relaxation, 
            "loss_lie_derivative": loss_lie_derivative,
            "loss_temporal": loss_temporal
        }
    
    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str
    ) -> torch.Tensor:
        """
        Shared logic for training, validation, and test steps (R3).

        Dynamics model and QP solver are sourced from self.dynamics_model /
        self.qp_solver (set by training.py before trainer.fit), NOT from the
        batch dict, which only ever contains states (C5).

        Three tiers:
          1. Both dynamics_model and qp_solver → full self-supervised CLF loss.
          2. Only dynamics_model → CLF positive-definiteness + Lie derivative loss.
          3. Neither → basic CLF positive-definiteness loss only.

        Args:
            batch: Batch dict from the dataloader (contains "states")
            stage: One of "train", "val", "test"

        Returns:
            Total scalar loss tensor
        """
        states = batch["states"]
        on_step = (stage == "train")

        dynamics_model = getattr(self, "dynamics_model", None)
        qp_solver = getattr(self, "qp_solver", None)

        if dynamics_model is not None and qp_solver is not None:
            next_states = batch.get("next_states", None)
            dt = batch.get("dt", 0.05)

            loss_dict = self.compute_self_supervised_clf_loss(
                states=states,
                dynamics_model=dynamics_model,
                qp_solver=qp_solver,
                next_states=next_states,
                dt=dt,
                alpha1=self.alpha1,
                alpha2=self.alpha2,
                alpha3=self.alpha3,
                alpha4=self.alpha4 if next_states is not None else 0.0
            )
            total_loss = loss_dict["loss"]
            self.log(f"{stage}_loss", total_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_loss_equilibrium", loss_dict["loss_equilibrium"], on_step=on_step, on_epoch=True, logger=True)
            self.log(f"{stage}_loss_relaxation", loss_dict["loss_relaxation"], on_step=on_step, on_epoch=True, logger=True)
            self.log(f"{stage}_loss_lie_derivative", loss_dict["loss_lie_derivative"], on_step=on_step, on_epoch=True, logger=True)
            if next_states is not None:
                self.log(f"{stage}_loss_temporal", loss_dict["loss_temporal"], on_step=on_step, on_epoch=True, logger=True)

        elif dynamics_model is not None:
            clf_loss = self.compute_clf_loss(states)
            lie_loss = self.compute_lie_derivative_loss(states, dynamics_model)
            total_loss = clf_loss + lie_loss
            self.log(f"{stage}_clf_loss", clf_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_lie_loss", lie_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_loss", total_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)

        else:
            total_loss = self.compute_clf_loss(states)
            self.log(f"{stage}_clf_loss", total_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_loss", total_loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step — delegates to _shared_step (R3)."""
        return {"loss": self._shared_step(batch, "train")}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step — delegates to _shared_step (R3)."""
        loss = self._shared_step(batch, "val")

        # On the first batch of each epoch, log visualizations
        if batch_idx == 0 and self.logger is not None and isinstance(self.logger, pl.loggers.WandbLogger):
            self._log_clf_visualizations(batch["states"])
            dynamics_model = getattr(self, "dynamics_model", None)
            if dynamics_model is not None:
                self._log_lie_derivative_visualizations(batch["states"], dynamics_model)

        return {"val_loss": loss}

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step — delegates to _shared_step (R3)."""
        return {"test_loss": self._shared_step(batch, "test")}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for PyTorch Lightning.
        
        Returns:
            Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _log_clf_visualizations(self, states: torch.Tensor) -> None:
        """
        Log CLF visualizations to Weights & Biases.

        Args:
            states: Batch of states [batch_size, state_dim]
        """
        # gradient() uses autograd.grad; use enable_grad() so this is safe
        # even if called from within a torch.no_grad() scope (C1).
        with torch.enable_grad():
            # Compute CLF values
            clf_values = self.compute_clf(states)
            
            # If state dimension is 2, create a 3D surface plot
            if self.state_dim == 2 and states.shape[0] >= 100:
                # Check if states form a grid (for better visualization)
                try:
                    # Reshape states into a grid for surface plot
                    grid_size = int(np.sqrt(states.shape[0]))
                    
                    if grid_size**2 == states.shape[0]:
                        # States are on a grid
                        x = states[:, 0].cpu().numpy().reshape(grid_size, grid_size)
                        y = states[:, 1].cpu().numpy().reshape(grid_size, grid_size)
                        z = clf_values.cpu().numpy().reshape(grid_size, grid_size)
                        
                        # Create 3D surface plot
                        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
                        fig.update_layout(
                            title="CLF Value Surface",
                            scene=dict(
                                xaxis_title="State Dimension 1",
                                yaxis_title="State Dimension 2",
                                zaxis_title="CLF Value"
                            )
                        )
                        
                        # Log to wandb
                        if isinstance(self.logger, pl.loggers.WandbLogger):
                            self.logger.experiment.log({"clf_surface": fig}, step=self.global_step)
                except:
                    # Not a grid, create scatter plot instead
                    pass
            
            # Create a scatter plot of CLF values vs. state norm
            state_norms = torch.norm(states, dim=1).cpu().numpy()
            clf_vals = clf_values.squeeze().cpu().numpy()
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=state_norms,
                    y=clf_vals,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=state_norms,
                        colorscale="Viridis",
                        showscale=True
                    )
                )
            )
            
            # Add reference line for positive definiteness (V(x) >= k*||x||²)
            x_ref = np.linspace(0, max(state_norms), 100)
            y_ref = 0.1 * x_ref**2  # Example reference function
            
            fig.add_trace(
                go.Scatter(
                    x=x_ref,
                    y=y_ref,
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Reference (0.1*||x||²)"
                )
            )
            
            fig.update_layout(
                title="CLF Values vs. State Norm",
                xaxis_title="State Norm ||x||",
                yaxis_title="CLF Value V(x)",
                showlegend=True
            )
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"clf_vs_norm": fig}, step=self.global_step)
            
            # Create gradient visualization
            if self.state_dim == 2:
                # Compute gradients
                gradients = self.gradient(states)
                
                # Create quiver plot for gradient field
                fig = go.Figure()
                
                # Add scatter plot for states
                fig.add_trace(
                    go.Scatter(
                        x=states[:, 0].cpu().numpy(),
                        y=states[:, 1].cpu().numpy(),
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=clf_values.squeeze().cpu().numpy(),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="CLF Value")
                        ),
                        name="States"
                    )
                )
                
                # Add quiver plot for gradients (subsample if there are too many points)
                max_arrows = 50
                if states.shape[0] > max_arrows:
                    stride = states.shape[0] // max_arrows
                    idx = np.arange(0, states.shape[0], stride)
                else:
                    idx = np.arange(states.shape[0])
                
                # Normalize gradients for better visualization
                grad_norms = torch.norm(gradients[idx], dim=1, keepdim=True)
                max_norm = grad_norms.max().item()
                if max_norm > 0:
                    normalized_gradients = gradients[idx] / max_norm
                else:
                    normalized_gradients = gradients[idx]
                
                # Add quiver plot
                fig.add_trace(
                    go.Scatter(
                        x=states[idx, 0].cpu().numpy(),
                        y=states[idx, 1].cpu().numpy(),
                        mode="markers+text",
                        marker=dict(
                            size=5,
                            color="rgba(0,0,0,0)"
                        ),
                        text=["↑" if g[1] > 0 else "↓" for g in normalized_gradients.detach().cpu().numpy()],
                        textposition="top center",
                        name="Gradient Direction"
                    )
                )
                
                fig.update_layout(
                    title="CLF Gradient Field",
                    xaxis_title="State Dimension 1",
                    yaxis_title="State Dimension 2"
                )
                
                # Log to wandb
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.experiment.log({"clf_gradient_field": fig}, step=self.global_step)
    
    def _log_lie_derivative_visualizations(self, states: torch.Tensor, dynamics_model: nn.Module) -> None:
        """
        Log Lie derivative visualizations to Weights & Biases.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            dynamics_model: Model of the system dynamics
        """
        # lie_derivatives → gradient() uses autograd.grad, which requires grad tracking.
        # Use enable_grad() so this method stays safe even if called from within
        # a torch.no_grad() scope (C1).
        with torch.enable_grad():
            # Compute CLF values and Lie derivatives
            clf_values = self.compute_clf(states)
            L_f_V, L_g_V = self.lie_derivatives(states, dynamics_model)

            # Compute state norms
            state_norms = torch.norm(states, dim=1).cpu().numpy()

            # Create scatter plot of L_f_V values
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=state_norms,
                    y=L_f_V.detach().squeeze().cpu().numpy(),
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=clf_values.detach().squeeze().cpu().numpy(),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="CLF Value")
                    ),
                    name="L_f_V"
                )
            )

            # Add zero reference line
            fig.add_hline(y=0, line=dict(color="red", dash="dash"), name="Zero")

            fig.update_layout(
                title="Lie Derivative L_f_V vs. State Norm",
                xaxis_title="State Norm ||x||",
                yaxis_title="L_f_V",
                showlegend=True
            )

            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"lie_derivative_lf": fig}, step=self.global_step)

            # Create scatter plot of L_g_V values (first dimension if action is multi-dimensional)
            fig = go.Figure()

            if L_g_V.shape[1] == 1:
                # Single action dimension
                fig.add_trace(
                    go.Scatter(
                        x=state_norms,
                        y=L_g_V.detach().squeeze().cpu().numpy(),
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=clf_values.detach().squeeze().cpu().numpy(),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="CLF Value")
                        ),
                        name="L_g_V"
                    )
                )

                fig.update_layout(
                    title="Lie Derivative L_g_V vs. State Norm",
                    xaxis_title="State Norm ||x||",
                    yaxis_title="L_g_V",
                    showlegend=True
                )
            else:
                # Multiple action dimensions, show norm of L_g_V
                L_g_V_norm = torch.norm(L_g_V, dim=1).detach().cpu().numpy()
                fig.add_trace(
                    go.Scatter(
                        x=state_norms,
                        y=L_g_V_norm,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=clf_values.detach().squeeze().cpu().numpy(),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="CLF Value")
                        ),
                        name="||L_g_V||"
                    )
                )

                fig.update_layout(
                    title="Norm of Lie Derivative L_g_V vs. State Norm",
                    xaxis_title="State Norm ||x||",
                    yaxis_title="||L_g_V||",
                    showlegend=True
                )

            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"lie_derivative_lg": fig}, step=self.global_step)

            # Stabilising control direction: u = -L_g_V / (||L_g_V|| + ε)  (R2)
            L_g_V_norm = torch.norm(L_g_V, dim=1, keepdim=True)
            epsilon = 1e-6
            u = -L_g_V / (L_g_V_norm + epsilon)

            # Compute Lie derivative with the computed control
            L_dot = L_f_V + torch.sum(L_g_V * u, dim=1, keepdim=True)

            # Create scatter plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=state_norms,
                    y=L_dot.detach().squeeze().cpu().numpy(),
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=clf_values.detach().squeeze().cpu().numpy(),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="CLF Value")
                    ),
                    name="L_dot with stabilizing control"
                )
            )

            # Add zero reference line
            fig.add_hline(y=0, line=dict(color="red", dash="dash"), name="Zero")

            fig.update_layout(
                title="Total Lie Derivative with Stabilizing Control vs. State Norm",
                xaxis_title="State Norm ||x||",
                yaxis_title="L_dot",
                showlegend=True
            )

            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"lie_derivative_total": fig}, step=self.global_step)
    
    def on_train_start(self) -> None:
        """
        Called when the training begins. Initialize tracking of CLF equilibrium value
        for monitoring during training.
        """
        super().on_train_start()
        # Initialize tracking of equilibrium value
        self.equilibrium_values = []
        self.global_steps = []
        
        # Store the grid for visualizations - we only need to create it once
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'config'):
            # Try to get environment name from wandb config
            env_name = self.logger.experiment.config.get('env_name', 'Pendulum-v1')
        else:
            # Default to Pendulum if not found
            env_name = 'Pendulum-v1'
            
        # Create the grid - cache for later use
        # Only call this once at the start of training for efficiency
        device = next(self.parameters()).device
        self.grid_states, self.grid_info = None, None
        
        try:
            from src.utils.clf_visualizations import create_state_grid_from_env
            self.grid_states, self.grid_info = create_state_grid_from_env(
                env_name=env_name,
                resolution=50,  # 50x50 grid for nice visualizations
                device=device
            )
            print(f"Created visualization grid for {env_name} with shape {self.grid_states.shape}")
        except Exception as e:
            print(f"Failed to create visualization grid: {e}")
            # Continue without the grid-based visualizations
    
    def on_train_batch_end(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """
        Called at the end of each training batch. Track CLF value at equilibrium.
        
        Args:
            outputs: Outputs from the training step
            batch: Current batch of data
            batch_idx: Index of the current batch
            dataloader_idx: Index of the current dataloader
        """
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
        
        # Log equilibrium value at regular intervals
        # Compute on device to avoid unnecessary transfers
        device = next(self.parameters()).device
        
        # Use the equilibrium buffer registered in __init__ (L6 — no hard-coded import)
        equilibrium_state = self.equilibrium
        
        # Compute CLF value at equilibrium
        with torch.no_grad():
            equilibrium_value = self.compute_clf(equilibrium_state.unsqueeze(0)).item()
        
        # Store values for plotting trends
        self.equilibrium_values.append(equilibrium_value)
        # Current global step is available from the trainer
        if hasattr(self, 'global_step'):
            self.global_steps.append(self.global_step)
        else:
            self.global_steps.append(len(self.equilibrium_values))
        
        # Log the equilibrium value
        self.log("clf_equilibrium_value", equilibrium_value, on_step=True, on_epoch=True, logger=True)
        
        # Every 100 steps, log a plot of the equilibrium value history
        if len(self.equilibrium_values) % 100 == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.global_steps,
                    y=self.equilibrium_values,
                    mode="lines",
                    name="CLF at Equilibrium"
                )
            )
            
            fig.update_layout(
                title="CLF Value at Equilibrium During Training",
                xaxis_title="Training Step",
                yaxis_title="CLF Value",
                showlegend=True
            )
            
            # Log to wandb
            wandb_logger = self.logger if isinstance(self.logger, pl.loggers.WandbLogger) else None
            if wandb_logger is not None:
                wandb_logger.experiment.log({"clf_equilibrium_history": fig}, step=self.global_step)
    
    def on_train_epoch_end(self) -> None:
        """
        Called at the end of each training epoch. Generate and log more detailed 
        visualizations using the grid.
        """
        super().on_train_epoch_end()
        
        # Only proceed if we have a grid and the necessary models
        if not hasattr(self, 'grid_states') or self.grid_states is None:
            return
        
        # Log the CLF value over the grid
        try:
            from src.utils.clf_visualizations import visualize_clf_value_grid
            
            # Generate contour plot of CLF values over the state space
            fig = visualize_clf_value_grid(self, self.grid_states, self.grid_info)
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"clf_value_contour": fig}, step=self.global_step)
        except Exception as e:
            print(f"Failed to generate CLF value contour: {e}")
        
        # After some epochs, generate more complex visualizations that require dynamics model
        # Only do this occasionally as it's computationally expensive
        if self.current_epoch % 5 == 0:  # Every 5 epochs
            self._log_admissible_control_visualizations()
    
    def _log_admissible_control_visualizations(self) -> None:
        """
        Generate and log visualizations of the admissible control set and Lie derivatives 
        with the infimum control.
        """
        # Check if we have necessary components
        if not hasattr(self, 'grid_states') or self.grid_states is None:
            return
        
        # Try to get dynamics model and action limits from the trainer
        dynamics_model = None
        action_bounds = (-5.0, 5.0)  # Default for pendulum
        action_dim = 1  # Default for pendulum
        
        # Get dynamics model from trainer's datamodule or callbacks
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'dynamics_model'):
            dynamics_model = self.trainer.datamodule.dynamics_model
        elif hasattr(self.trainer, 'callbacks'):
            for callback in self.trainer.callbacks:
                if hasattr(callback, 'dynamics_model'):
                    dynamics_model = callback.dynamics_model
                    break
        
        # Get action bounds from trainer or config
        try:
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'env'):
                # Get from environment
                env = self.trainer.datamodule.env
                action_bounds = (float(env.action_space.low[0]), float(env.action_space.high[0]))
                action_dim = env.action_space.shape[0]
            elif hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'config'):
                # Try to get from wandb config
                cfg = self.logger.experiment.config
                if 'clf' in cfg and 'qp_solver' in cfg['clf'] and 'action_limits' in cfg['clf']['qp_solver']:
                    action_bounds = tuple(cfg['clf']['qp_solver']['action_limits'])
                if 'action_dim' in cfg:
                    action_dim = cfg['action_dim']
        except Exception as e:
            print(f"Failed to get action bounds: {e}")
        
        # Check if we have all necessary components
        if dynamics_model is None:
            print("No dynamics model available for admissible control visualization")
            return
        
        # Generate admissible control set visualization
        try:
            from src.utils.clf_visualizations import visualize_admissible_control_set
            
            # Generate contour plots
            fig, infimum_actions, lie_derivatives = visualize_admissible_control_set(
                clf_model=self,
                dynamics_model=dynamics_model,
                grid_states=self.grid_states,
                grid_info=self.grid_info,
                action_bounds=action_bounds,
                action_dim=action_dim,
                resolution=100  # Resolution for action sampling
            )
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"admissible_control_set": fig}, step=self.global_step)

                # Also log statistics about the admissible control set
                n_admissible = torch.sum(infimum_actions != float('inf')).item()
                total_points = len(infimum_actions)
                coverage = 100.0 * n_admissible / total_points

                # Log the percentage of states that have an admissible control
                self.logger.experiment.log({
                    "admissible_control_coverage": coverage,
                    "admissible_control_count": n_admissible,
                }, step=self.global_step)

                # Log histogram of infimum action values (excluding points with no admissible control)
                valid_actions = infimum_actions[infimum_actions != float('inf')].cpu().numpy()
                if len(valid_actions) > 0:
                    fig = go.Figure(data=go.Histogram(x=valid_actions, nbinsx=30))
                    fig.update_layout(
                        title="Distribution of Infimum Admissible Controls",
                        xaxis_title="Control Value",
                        yaxis_title="Count"
                    )
                    self.logger.experiment.log({"admissible_control_histogram": fig}, step=self.global_step)
                
        except Exception as e:
            print(f"Failed to generate admissible control visualization: {e}")
            import traceback
            traceback.print_exc()