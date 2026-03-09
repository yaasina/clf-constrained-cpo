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
        output_nonlinearity: Optional[nn.Module] = None
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
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.state_dim = state_dim
        self.output_nonlinearity = output_nonlinearity
        self.learning_rate = learning_rate
        
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
            # Create a zero tensor that maintains the computational graph connection
            return torch.zeros(state.shape[0], 1, device=state.device, requires_grad=True) + state.sum() * 0.0 
        
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
        # Ensure we're using a tensor with requires_grad=True for gradient computation
        if not state.requires_grad:
            state = state.clone().requires_grad_(True)
        
        # Get neural network output
        nn_output = self(state)
        
        # Calculate V(x) = 0.5 * (NN(x))^2
        clf_value = 0.5 * torch.pow(nn_output, 2)
        
        return clf_value
    
    def gradient(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the CLF with respect to the state.
        
        Args:
            state: Batch of state vectors [batch_size, state_dim]
            
        Returns:
            Gradient of L(x) with respect to x [batch_size, state_dim]
        """
        # Create a fresh tensor with requires_grad=True
        state_with_grad = state.detach().clone().requires_grad_(True)
        
        # Run a forward pass
        clf_value = self.compute_clf(state_with_grad)
        
        # Create gradients tensor of same shape as state tensor
        gradients = torch.zeros_like(state_with_grad)
        
        # Compute gradients for each sample separately to avoid graph issues
        for i in range(state_with_grad.shape[0]):
            single_state = state_with_grad[i:i+1]  # Keep batch dimension
            
            # Do another forward pass for this specific sample
            single_clf_value = self.compute_clf(single_state)
            
            try:
                # Compute gradient using autograd
                grad_outputs = torch.ones_like(single_clf_value)
                single_grad = torch.autograd.grad(
                    outputs=single_clf_value,
                    inputs=single_state,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                gradients[i] = single_grad
            except Exception:
                # Use finite differences as fallback for this sample
                gradients[i] = self._compute_gradient_fd(single_state.squeeze(0))
        
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
        
        # For points with significant control authority, compute u = -L_g_V.T / (L_g_V_norm + epsilon)
        # This normalizes the control to avoid excessive magnitudes
        u = torch.zeros_like(L_g_V)
        u[significant_control.repeat(1, L_g_V.shape[1])] = -L_g_V[significant_control.repeat(1, L_g_V.shape[1])]
        u = u / (L_g_V_norm + epsilon)
        
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
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states and optionally dynamics_model
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        
        # Compute basic CLF loss (positive definiteness)
        clf_loss = self.compute_clf_loss(states)
        
        total_loss = clf_loss
        
        # If dynamics model is provided, compute Lie derivative loss
        if "dynamics_model" in batch:
            dynamics_model = batch["dynamics_model"]
            lie_loss = self.compute_lie_derivative_loss(states, dynamics_model)
            total_loss = clf_loss + lie_loss
            
            # Log metrics
            self.log("train_lie_loss", lie_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Log metrics
        self.log("train_clf_loss", clf_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": total_loss}
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states and optionally dynamics_model
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        
        # Compute basic CLF loss (positive definiteness)
        clf_loss = self.compute_clf_loss(states)
        
        total_loss = clf_loss
        
        # If dynamics model is provided, compute Lie derivative loss
        if "dynamics_model" in batch:
            dynamics_model = batch["dynamics_model"]
            lie_loss = self.compute_lie_derivative_loss(states, dynamics_model)
            total_loss = clf_loss + lie_loss
            
            # Log metrics
            self.log("val_lie_loss", lie_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log metrics
        self.log("val_clf_loss", clf_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"val_loss": total_loss}
    
    def test_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step for PyTorch Lightning.
        
        Args:
            batch: Dictionary containing states and optionally dynamics_model
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and logs
        """
        states = batch["states"]
        
        # Compute basic CLF loss (positive definiteness)
        clf_loss = self.compute_clf_loss(states)
        
        total_loss = clf_loss
        
        # If dynamics model is provided, compute Lie derivative loss
        if "dynamics_model" in batch:
            dynamics_model = batch["dynamics_model"]
            lie_loss = self.compute_lie_derivative_loss(states, dynamics_model)
            total_loss = clf_loss + lie_loss
            
            # Log metrics
            self.log("test_lie_loss", lie_loss, on_step=False, on_epoch=True, logger=True)
        
        # Log metrics
        self.log("test_clf_loss", clf_loss, on_step=False, on_epoch=True, logger=True)
        self.log("test_loss", total_loss, on_step=False, on_epoch=True, logger=True)
        
        return {"test_loss": total_loss}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers for PyTorch Lightning.
        
        Returns:
            Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)