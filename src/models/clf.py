"""
Module for Control Lyapunov Function (CLF) models.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go


class CLFNetwork(nn.Module):
    """
    Neural network model for learning a Control Lyapunov Function (CLF).

    V(x) = (x-x*)^T P (x-x*) + ||v_θ(x) - v_θ(x*)||²

    P = A^T A + eps_pd * I  is positive definite by construction (A is a learnable
    n×n matrix, eps_pd > 0), which prevents gradient collapse and ensures:
      - V(x) > 0 for all x ≠ x*
      - V(x*) = 0 exactly (no equilibrium loss term needed)
      - ∇V has a quadratic lower bound, keeping Lie derivatives well-conditioned
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
        equilibrium: Optional[torch.Tensor] = None,
        exp_const: float = 1.0,
        eps_pd: float = 1e-2,
        residual_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize the CLF network.

        V(x) = (x-x*)^T P (x-x*) + ||v_θ(x) - v_θ(x*)||²

        P = A^T A + eps_pd * I  is positive definite by construction.
        v_θ(x) is the vector output of the residual network (size residual_dim,
        defaults to state_dim).

        Args:
            state_dim: Dimension of the state space
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout probability for regularization
            learning_rate: Learning rate for optimizer
            nonlinearity: Activation function for hidden layers
            output_nonlinearity: Activation function for output layer (None for linear output)
            loss: Dictionary of loss hyperparameters (alpha1, alpha2, alpha3, alpha4)
            equilibrium: Equilibrium state tensor [state_dim]. Defaults to the zero vector.
            exp_const: Exponential decay constant μ for the CLF condition
                       L_f_V + L_g_V·u + μ·V(x) ≤ 0 (must be positive for exponential stability).
            eps_pd: Lower bound on eigenvalues of P (ensures strict positive definiteness).
            residual_dim: Output dimension of v_θ(x). Defaults to state_dim if None.
        """
        super().__init__()

        self.state_dim = state_dim
        self.residual_dim = residual_dim if residual_dim is not None else state_dim
        self.output_nonlinearity = output_nonlinearity
        self.learning_rate = learning_rate
        self.eps_pd = eps_pd

        # Set loss hyperparameters with defaults
        self.loss_params = loss or {}
        self.alpha1 = self.loss_params.get('alpha1', 1.0)
        self.alpha2 = self.loss_params.get('alpha2', 0.1)
        self.alpha3 = self.loss_params.get('alpha3', 1.0)
        self.alpha4 = self.loss_params.get('alpha4', 1.0)
        self.exp_const = exp_const

        # Register equilibrium as a buffer so it moves to the right device automatically
        # and is saved / restored with the checkpoint (L6)
        _eq = equilibrium.clone().detach().float() if equilibrium is not None else torch.zeros(state_dim)
        self.register_buffer("equilibrium", _eq)

        # Learnable n×n matrix for the quadratic term: P = A^T A + eps_pd * I
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)

        # Residual network: outputs a residual_dim-dimensional vector v_θ(x)
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
            nn.Linear(hidden_dim, self.residual_dim)
        )

        # Tracking state for the custom training loop
        self.current_epoch: int = 0
        self.global_step: int = 0
        self._wandb_run = None  # set by Trainer before fit()

        # History tracking for visualizations
        self.equilibrium_values: List[float] = []
        self.global_steps: List[int] = []

        # Store hparams for checkpoint reconstruction
        self._hparams = {
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "exp_const": exp_const,
            "eps_pd": eps_pd,
            "residual_dim": self.residual_dim,
            "loss": loss,
        }

    def _log_metric(self, key: str, val, **kwargs) -> None:
        """Log a scalar metric to W&B if a run is available."""
        if self._wandb_run is not None:
            v = val.item() if torch.is_tensor(val) else float(val)
            self._wandb_run.log({key: v}, step=self.global_step)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute v_θ(x), the residual vector.

        Args:
            state: Batch of state vectors [batch_size, state_dim]

        Returns:
            Residual vector [batch_size, state_dim]
        """
        output = self.layers(state)

        if self.output_nonlinearity is not None:
            output = self.output_nonlinearity(output)

        return output

    def compute_clf(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the CLF value:
            V(x) = (x-x*)^T P (x-x*) + ||v_θ(x) - v_θ(x*)||²

        P = A^T A + eps_pd * I  is positive definite by construction,
        so V(x) > 0 for all x ≠ x* and V(x*) = 0 exactly.

        Args:
            state: Batch of state vectors [batch_size, state_dim]

        Returns:
            CLF values [batch_size, 1]
        """
        device = state.device
        n = self.state_dim

        # Quadratic term: (x - x*)^T P (x - x*)
        P = self.A.T @ self.A + self.eps_pd * torch.eye(n, device=device, dtype=state.dtype)
        deviation = state - self.equilibrium          # [B, n]
        Pd = deviation @ P                            # [B, n]
        quad = (Pd * deviation).sum(dim=1, keepdim=True)  # [B, 1]

        # Residual term: ||v_θ(x) - v_θ(x*)||²
        # v_eq is treated as constant w.r.t. state (equilibrium is a fixed buffer)
        v_x = self(state)                                          # [B, n]
        v_eq = self(self.equilibrium.unsqueeze(0))                 # [1, n]
        residual = ((v_x - v_eq) ** 2).sum(dim=1, keepdim=True)   # [B, 1]

        return quad + residual

    def gradient(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the CLF with respect to the state.

        Uses a single vectorised autograd call over the entire batch:
        d(ΣᵢV(xᵢ))/d(xⱼ) = dV(xⱼ)/d(xⱼ) because the network has no
        cross-sample interactions.

        Args:
            state: Batch of state vectors [batch_size, state_dim]

        Returns:
            Gradient of V(x) with respect to x [batch_size, state_dim]
        """
        state_grad = state.detach().clone().requires_grad_(True)
        # enable_grad ensures this works even when called from validation_step
        # (no_grad context). create_graph is only needed during training.  (S2)
        with torch.enable_grad():
            clf_sum = self.compute_clf(state_grad).sum()
            gradients = torch.autograd.grad(
                clf_sum, state_grad, create_graph=self.training
            )[0]
        return gradients

    def lie_derivatives(
        self,
        state: torch.Tensor,
        dynamics_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Lie derivatives of the CLF with respect to the dynamics.

        Args:
            state: Batch of state vectors [batch_size, state_dim]
            dynamics_model: Model of the system dynamics

        Returns:
            L_f_V: Lie derivative along f [batch_size, 1]
            L_g_V: Lie derivative along g [batch_size, action_dim]
        """
        grad_L = self.gradient(state)

        try:
            with torch.no_grad():
                try:
                    f_x, g_x = dynamics_model(state)
                except Exception as e:
                    if hasattr(dynamics_model, 'models') and len(dynamics_model.models) > 0:
                        f_x, g_x = dynamics_model.models[0](state)
                    elif hasattr(dynamics_model, 'f') and hasattr(dynamics_model, 'g'):
                        f_x = dynamics_model.f(state)
                        g_x = dynamics_model.g(state)
                    else:
                        raise ValueError(f"Failed to get dynamics: {e}")
        except Exception as e:
            print(f"Error getting dynamics for Lie derivative: {e}")
            raise

        L_f_V = torch.sum(grad_L * f_x, dim=1, keepdim=True)

        if g_x.dim() == 3:  # [batch_size, state_dim, action_dim]
            L_g_V = torch.bmm(grad_L.unsqueeze(1), g_x).squeeze(1)
        else:
            try:
                action_dim = g_x.shape[1] // state.shape[1]
                g_x_reshaped = g_x.reshape(g_x.shape[0], state.shape[1], action_dim)
                L_g_V = torch.bmm(grad_L.unsqueeze(1), g_x_reshaped).squeeze(1)
            except Exception as e:
                print(f"Error reshaping g_x. Shapes: state={state.shape}, g_x={g_x.shape}")
                raise e

        return L_f_V, L_g_V

    def compute_lie_derivative_with_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dynamics_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute L_f_V + L_g_V·u + exp_const·V(x).

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            dynamics_model: system dynamics

        Returns:
            CLF condition value [batch_size, 1]
        """
        L_f_V, L_g_V = self.lie_derivatives(state, dynamics_model)
        L_g_V_u = torch.sum(L_g_V * action, dim=1, keepdim=True)
        V = self.compute_clf(state)
        return L_f_V + L_g_V_u + self.exp_const * V

    def compute_self_supervised_clf_loss(
        self,
        states: torch.Tensor,
        dynamics_model: nn.Module,
        qp_solver,
        next_states: Optional[torch.Tensor] = None,
        dt: float = 0.05,
        alpha1: float = 1.0,
        alpha2: float = 0.1,
        alpha3: float = 1.0,
        alpha4: float = 1.0,
        exp_const: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the self-supervised CLF loss:
        L = α₁·V(eq) + (α₂/B)Σrᵢ + (α₃/B)Σmax(L_f V + L_g V·u + μ·V, 0)
            + (α₄/B)Σmax((V(x⁺)-V(x))/Δt, 0)
        """
        batch_size = states.shape[0]
        device = states.device

        # Term 1: V(eq) should be 0 (L6 — uses registered buffer, no hard-coded import)
        equilibrium_value = self.compute_clf(self.equilibrium.to(device).unsqueeze(0))
        loss_equilibrium = alpha1 * equilibrium_value

        # Compute CLF values once — reused in Term 3 and Term 4
        clf_values = self.compute_clf(states)

        # Compute Lie derivatives once — reused in Term 3 (G5)
        L_f_V, L_g_V = self.lie_derivatives(states, dynamics_model)

        # Solve QP for optimal controls and relaxation variables
        qp_results = qp_solver.solve_batch(states, self, dynamics_model, batch_size=batch_size)
        u_values = qp_results['u_values'].to(device)
        r_values = qp_results['r_values'].to(device)

        # Term 2: minimise relaxation variables; filter inf from failed QP solves (L7)
        valid_r = r_values[~torch.isinf(r_values)]
        if valid_r.numel() > 0:
            loss_relaxation = alpha2 * valid_r.mean()
        else:
            loss_relaxation = torch.tensor(0.0, device=device, requires_grad=False)

        # Term 3: CLF decrease condition
        if u_values.dim() == 1:
            u_values = u_values.unsqueeze(-1)
        L_dot = L_f_V + torch.sum(L_g_V * u_values, dim=1, keepdim=True) + exp_const * clf_values
        loss_lie_derivative = alpha3 * torch.relu(L_dot).mean()

        # Term 4: temporal decrease (only if next_states provided)
        loss_temporal = torch.tensor(0.0, device=device)
        if next_states is not None:
            next_values = self.compute_clf(next_states)
            v_dot = (next_values - clf_values) / dt
            loss_temporal = alpha4 * torch.mean(torch.relu(v_dot))
        
        # Term 5: Regularize CLF scale to be close to state norm squared
        state_norm_sq = torch.sum(states.pow(2), dim=1, keepdim=True)
        loss_scaling = 0.1 * torch.mean((clf_values - state_norm_sq).pow(2))

        total_loss = loss_equilibrium + loss_relaxation + loss_lie_derivative + loss_temporal + loss_scaling

        return {
            "loss": total_loss,
            "loss_equilibrium": loss_equilibrium,
            "loss_relaxation": loss_relaxation,
            "loss_lie_derivative": loss_lie_derivative,
            "loss_temporal": loss_temporal,
            "loss_scaling": loss_scaling,
        }

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str
    ) -> torch.Tensor:
        """Shared logic for training, validation, and test steps."""
        states = batch["states"]
        next_states = batch.get("next_states", None)
        dt = batch.get("dt", 0.05)

        dynamics_model = getattr(self, "dynamics_model", None)
        qp_solver = getattr(self, "qp_solver", None)
        if dynamics_model is None or qp_solver is None:
            raise RuntimeError(
                "CLFNetwork._shared_step requires both dynamics_model and qp_solver. "
                "Set them via object.__setattr__(model, 'dynamics_model', ...) before training."
            )

        loss_dict = self.compute_self_supervised_clf_loss(
            states=states,
            dynamics_model=dynamics_model,
            qp_solver=qp_solver,
            next_states=next_states,
            dt=dt,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            alpha3=self.alpha3,
            alpha4=self.alpha4 if next_states is not None else 0.0,
            exp_const=self.exp_const,
        )
        total_loss = loss_dict["loss"]
        self._log_metric(f"{stage}_loss", total_loss)
        self._log_metric(f"{stage}_loss_equilibrium", loss_dict["loss_equilibrium"])
        self._log_metric(f"{stage}_loss_relaxation", loss_dict["loss_relaxation"])
        self._log_metric(f"{stage}_loss_lie_derivative", loss_dict["loss_lie_derivative"])
        if next_states is not None:
            self._log_metric(f"{stage}_loss_temporal", loss_dict["loss_temporal"])
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step — delegates to _shared_step (R3)."""
        return {"loss": self._shared_step(batch, "train")}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step — delegates to _shared_step (R3)."""
        loss = self._shared_step(batch, "val")

        if batch_idx == 0 and self._wandb_run is not None:
            self._log_clf_visualizations(batch["states"])
            dynamics_model = getattr(self, "dynamics_model", None)
            if dynamics_model is not None:
                self._log_lie_derivative_visualizations(batch["states"], dynamics_model)

        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step — delegates to _shared_step (R3)."""
        return {"test_loss": self._shared_step(batch, "test")}

    # ── Training hooks ──────────────────────────────────────────────────────────

    def on_train_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Called at the end of each training batch. Track CLF value at equilibrium. (S1)"""
        device = next(self.parameters()).device

        # Use the equilibrium buffer (L6 — no hard-coded import)
        equilibrium_state = self.equilibrium

        # Temporarily switch to eval so gradient() uses create_graph=False (monitoring only)
        self.eval()
        with torch.no_grad():
            equilibrium_value = self.compute_clf(equilibrium_state.unsqueeze(0)).item()

        # Flatness monitors: mean CLF magnitude and mean gradient norm over the batch
        states = batch.get("states")
        if states is not None:
            with torch.no_grad():
                clf_vals = self.compute_clf(states)
                clf_magnitude = clf_vals.mean().item()
            grad_L = self.gradient(states)  # enable_grad is used internally; create_graph=False in eval
            grad_norm_mean = torch.norm(grad_L.detach(), dim=1).mean().item()
        else:
            clf_magnitude = float("nan")
            grad_norm_mean = float("nan")
        self.train()

        self.equilibrium_values.append(equilibrium_value)
        self.global_steps.append(self.global_step)

        self._log_metric("clf_equilibrium_value", equilibrium_value)
        self._log_metric("clf_magnitude_mean", clf_magnitude)
        self._log_metric("clf_grad_norm_mean", grad_norm_mean)

        # Every 100 steps, log a plot of equilibrium value history
        if len(self.equilibrium_values) % 100 == 0 and self._wandb_run is not None:
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
            self._wandb_run.log({"clf_equilibrium_history": fig}, step=self.global_step)

    # ── Visualization helpers ───────────────────────────────────────────────────

    def _log_clf_visualizations(self, states: torch.Tensor) -> None:
        """Log CLF visualizations to W&B."""
        # gradient() uses autograd.grad; enable_grad() keeps this safe in no_grad context (C1)
        with torch.enable_grad():
            # Detach for visualization only — no backprop needed (S3)
            clf_values = self.compute_clf(states).detach()

            state_norms = torch.norm(states, dim=1).cpu().numpy()
            clf_vals = clf_values.squeeze().cpu().numpy()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=state_norms,
                    y=clf_vals,
                    mode="markers",
                    marker=dict(size=8, color=state_norms, colorscale="Viridis", showscale=True)
                )
            )
            x_ref = np.linspace(0, max(state_norms) if len(state_norms) > 0 else 1.0, 100)
            y_ref = 0.1 * x_ref**2
            fig.add_trace(
                go.Scatter(
                    x=x_ref, y=y_ref,
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
            if self._wandb_run is not None:
                self._wandb_run.log({"clf_vs_norm": fig}, step=self.global_step)

    def _log_lie_derivative_visualizations(self, states: torch.Tensor, dynamics_model: nn.Module) -> None:
        """Log Lie derivative visualizations to W&B."""
        # lie_derivatives → gradient() uses autograd.grad (C1)
        with torch.enable_grad():
            clf_values = self.compute_clf(states)
            L_f_V, L_g_V = self.lie_derivatives(states, dynamics_model)

            state_norms = torch.norm(states, dim=1).cpu().numpy()

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
            fig.add_hline(y=0, line=dict(color="red", dash="dash"), name="Zero")
            fig.update_layout(
                title="Lie Derivative L_f_V vs. State Norm",
                xaxis_title="State Norm ||x||",
                yaxis_title="L_f_V",
                showlegend=True
            )
            if self._wandb_run is not None:
                self._wandb_run.log({"lie_derivative_lf": fig}, step=self.global_step)

            fig = go.Figure()
            if L_g_V.shape[1] == 1:
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
            if self._wandb_run is not None:
                self._wandb_run.log({"lie_derivative_lg": fig}, step=self.global_step)

            # Stabilising control direction: u = -L_g_V / (||L_g_V|| + ε)  (R2)
            L_g_V_norm = torch.norm(L_g_V, dim=1, keepdim=True)
            epsilon = 1e-6
            u = -L_g_V / (L_g_V_norm + epsilon)
            L_dot = L_f_V + torch.sum(L_g_V * u, dim=1, keepdim=True)

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
            fig.add_hline(y=0, line=dict(color="red", dash="dash"), name="Zero")
            fig.update_layout(
                title="Total Lie Derivative with Stabilizing Control vs. State Norm",
                xaxis_title="State Norm ||x||",
                yaxis_title="L_dot",
                showlegend=True
            )
            if self._wandb_run is not None:
                self._wandb_run.log({"lie_derivative_total": fig}, step=self.global_step)

    def save_checkpoint(self, path: str, val_loss: Optional[float] = None) -> None:
        """Save model to a checkpoint file."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "hparams": self._hparams,
            "equilibrium": self.equilibrium.cpu(),
            "val_loss": val_loss,
            "epoch": self.current_epoch,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = "cpu") -> "CLFNetwork":
        """Load model from a checkpoint file saved by save_checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)
        hparams = checkpoint["hparams"]
        eq = checkpoint.get("equilibrium", None)
        model = cls(**hparams, equilibrium=eq)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# Backward-compatibility alias
CLFNetworkLightning = CLFNetwork
