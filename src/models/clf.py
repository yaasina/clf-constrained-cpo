"""
Module for Control Lyapunov Function (CLF) models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CLFNetwork(nn.Module):
    """
    Neural network model for learning a Control Lyapunov Function (CLF).
    The CLF is of the form V(x) = 0.5 * (NN(x))^2, which ensures V >= 0
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
            exp_const: Exponential decay constant μ for the CLF condition
                       L_f_V + L_g_V·u + μ·V(x) ≤ 0 (must be positive for exponential stability).
        """
        super().__init__()

        self.state_dim = state_dim
        self.output_nonlinearity = output_nonlinearity
        self.learning_rate = learning_rate

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
            nn.Linear(hidden_dim, 1)
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
            "loss": loss,
        }

    def _log_metric(self, key: str, val, **kwargs) -> None:
        """Log a scalar metric to W&B if a run is available."""
        if self._wandb_run is not None:
            v = val.item() if torch.is_tensor(val) else float(val)
            self._wandb_run.log({key: v}, step=self.global_step)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute NN(x).

        Args:
            state: Batch of state vectors [batch_size, state_dim]

        Returns:
            Network output [batch_size, 1]
        """


        output = self.layers(state)

        if self.output_nonlinearity is not None:
            output = self.output_nonlinearity(output)

        return output

    def compute_clf(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the CLF value V(x) = 0.5 * (NN(x))^2.

        Args:
            state: Batch of state vectors [batch_size, state_dim]

        Returns:
            CLF values [batch_size, 1]
        """
        nn_output = self(state)
        return 0.5 * torch.pow(nn_output, 2)

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

    def _compute_gradient_fd(self, state: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute gradient using finite differences as a fallback."""
        state_np = state.detach().cpu().numpy()
        state_dim = state_np.shape[0]
        gradient = np.zeros(state_dim)
        base_state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_value = self.compute_clf(base_state).item()
        for i in range(state_dim):
            perturbed_state_np = state_np.copy()
            perturbed_state_np[i] += epsilon
            perturbed_state = torch.tensor(perturbed_state_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                perturbed_value = self.compute_clf(perturbed_state).item()
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

    def compute_clf_loss(
        self,
        states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        clf_values = self.compute_clf(states)

        if targets is not None:
            loss = F.mse_loss(clf_values, targets)
        else:
            state_norms = torch.norm(states, dim=1, keepdim=True)
            zero_mask = state_norms < epsilon
            target_values = 0.5 * state_norms**2
            target_values[zero_mask] = 0.0
            loss = F.mse_loss(clf_values, target_values)
            # Add penalty for negative CLF values to encourage non-negativity
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
        # Mask points very close to the equilibrium
        state_norms = torch.norm(states, dim=1, keepdim=True)
        non_zero_mask = state_norms > epsilon

        if not torch.any(non_zero_mask):
            return torch.tensor(0.0, device=states.device, requires_grad=True)

        L_f_V, L_g_V = self.lie_derivatives(states, dynamics_model)

        L_g_V_norm = torch.norm(L_g_V, dim=1, keepdim=True)
        # Normalize control direction without boolean indexing (R2)
        u = -L_g_V / (L_g_V_norm + epsilon)

        L_dot = L_f_V + torch.sum(L_g_V * u, dim=1, keepdim=True)

        alpha = 0.1
        clf_values = self.compute_clf(states)
        desired_L_dot = -alpha * clf_values
        lie_loss = torch.relu(L_dot - desired_L_dot).mean()

        return lambda_lie * lie_loss

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
        u_values = qp_results['u_values']
        r_values = qp_results['r_values']

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

        total_loss = loss_equilibrium + loss_relaxation + loss_lie_derivative + loss_temporal

        return {
            "loss": total_loss,
            "loss_equilibrium": loss_equilibrium,
            "loss_relaxation": loss_relaxation,
            "loss_lie_derivative": loss_lie_derivative,
            "loss_temporal": loss_temporal,
        }

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str
    ) -> torch.Tensor:
        """
        Shared logic for training, validation, and test steps (R3).

        Three tiers:
          1. Both dynamics_model and qp_solver → full self-supervised CLF loss.
          2. Only dynamics_model → CLF PD + Lie derivative loss.
          3. Neither → basic CLF PD loss.
        """
        states = batch["states"]

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

        elif dynamics_model is not None:
            clf_loss = self.compute_clf_loss(states)
            lie_loss = self.compute_lie_derivative_loss(states, dynamics_model)
            total_loss = clf_loss + lie_loss
            self._log_metric(f"{stage}_clf_loss", clf_loss)
            self._log_metric(f"{stage}_lie_loss", lie_loss)
            self._log_metric(f"{stage}_loss", total_loss)

        else:
            total_loss = self.compute_clf_loss(states)
            self._log_metric(f"{stage}_clf_loss", total_loss)
            self._log_metric(f"{stage}_loss", total_loss)

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

    def on_train_start(self) -> None:
        """Called when training begins. Initialize visualization grid."""
        self.equilibrium_values = []
        self.global_steps = []

        # Default to Pendulum environment
        env_name = 'Pendulum-v1'

        device = next(self.parameters()).device
        self.grid_states, self.grid_info = None, None

        try:
            from src.utils.clf_visualizations import create_state_grid_from_env
            self.grid_states, self.grid_info = create_state_grid_from_env(
                env_name=env_name,
                resolution=50,
                device=device
            )
            print(f"Created visualization grid for {env_name} with shape {self.grid_states.shape}")
        except Exception as e:
            print(f"Failed to create visualization grid: {e}")

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

        with torch.no_grad():
            equilibrium_value = self.compute_clf(equilibrium_state.unsqueeze(0)).item()

        self.equilibrium_values.append(equilibrium_value)
        self.global_steps.append(self.global_step)

        self._log_metric("clf_equilibrium_value", equilibrium_value)

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

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch. Log CLF value grid contour."""
        if not hasattr(self, 'grid_states') or self.grid_states is None:
            return

        try:
            from src.utils.clf_visualizations import visualize_clf_value_grid
            fig = visualize_clf_value_grid(self, self.grid_states, self.grid_info)
            if self._wandb_run is not None:
                self._wandb_run.log({"clf_value_contour": fig}, step=self.global_step)
        except Exception as e:
            print(f"Failed to generate CLF value contour: {e}")

        if self.current_epoch % 5 == 0:
            self._log_admissible_control_visualizations()

    # ── Visualization helpers ───────────────────────────────────────────────────

    def _log_clf_visualizations(self, states: torch.Tensor) -> None:
        """Log CLF visualizations to W&B."""
        # gradient() uses autograd.grad; enable_grad() keeps this safe in no_grad context (C1)
        with torch.enable_grad():
            # Detach for visualization only — no backprop needed (S3)
            clf_values = self.compute_clf(states).detach()

            if self.state_dim == 2 and states.shape[0] >= 100:
                try:
                    grid_size = int(np.sqrt(states.shape[0]))
                    if grid_size**2 == states.shape[0]:
                        x = states[:, 0].cpu().numpy().reshape(grid_size, grid_size)
                        y = states[:, 1].cpu().numpy().reshape(grid_size, grid_size)
                        z = clf_values.cpu().numpy().reshape(grid_size, grid_size)
                        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
                        fig.update_layout(
                            title="CLF Value Surface",
                            scene=dict(
                                xaxis_title="State Dimension 1",
                                yaxis_title="State Dimension 2",
                                zaxis_title="CLF Value"
                            )
                        )
                        if self._wandb_run is not None:
                            self._wandb_run.log({"clf_surface": fig}, step=self.global_step)
                except Exception:
                    pass

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

            if self.state_dim == 2:
                gradients = self.gradient(states)
                fig = go.Figure()
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
                max_arrows = 50
                if states.shape[0] > max_arrows:
                    stride = states.shape[0] // max_arrows
                    idx = np.arange(0, states.shape[0], stride)
                else:
                    idx = np.arange(states.shape[0])
                grad_norms = torch.norm(gradients[idx], dim=1, keepdim=True)
                max_norm = grad_norms.max().item()
                normalized_gradients = gradients[idx] / max_norm if max_norm > 0 else gradients[idx]
                fig.add_trace(
                    go.Scatter(
                        x=states[idx, 0].cpu().numpy(),
                        y=states[idx, 1].cpu().numpy(),
                        mode="markers+text",
                        marker=dict(size=5, color="rgba(0,0,0,0)"),
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
                if self._wandb_run is not None:
                    self._wandb_run.log({"clf_gradient_field": fig}, step=self.global_step)

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

    def _log_admissible_control_visualizations(self) -> None:
        """Generate and log visualizations of the admissible control set."""
        if not hasattr(self, 'grid_states') or self.grid_states is None:
            return

        # Skip expensive computation if there's nowhere to log
        if self._wandb_run is None:
            return

        # Get dynamics model from self (was attached by training.py before fit)
        dynamics_model = getattr(self, 'dynamics_model', None)
        if dynamics_model is None:
            print("No dynamics model available for admissible control visualization")
            return

        action_bounds = (-5.0, 5.0)
        action_dim = 1

        # Try to get action bounds from attached qp_solver
        qp_solver = getattr(self, 'qp_solver', None)
        if qp_solver is not None:
            action_bounds = (qp_solver.action_lower, qp_solver.action_upper)
            if qp_solver.action_dim is not None:
                action_dim = qp_solver.action_dim

        try:
            from src.utils.clf_visualizations import visualize_admissible_control_set

            fig, infimum_actions, lie_derivatives = visualize_admissible_control_set(
                clf_model=self,
                dynamics_model=dynamics_model,
                grid_states=self.grid_states,
                grid_info=self.grid_info,
                action_bounds=action_bounds,
                action_dim=action_dim,
                resolution=100
            )

            if self._wandb_run is not None:
                self._wandb_run.log({"admissible_control_set": fig}, step=self.global_step)

                n_admissible = torch.sum(infimum_actions != float('inf')).item()
                total_points = len(infimum_actions)
                coverage = 100.0 * n_admissible / total_points

                self._wandb_run.log({
                    "admissible_control_coverage": coverage,
                    "admissible_control_count": n_admissible,
                }, step=self.global_step)

                valid_actions = infimum_actions[infimum_actions != float('inf')].cpu().numpy()
                if len(valid_actions) > 0:
                    fig = go.Figure(data=go.Histogram(x=valid_actions, nbinsx=30))
                    fig.update_layout(
                        title="Distribution of Infimum Admissible Controls",
                        xaxis_title="Control Value",
                        yaxis_title="Count"
                    )
                    self._wandb_run.log({"admissible_control_histogram": fig}, step=self.global_step)

        except Exception as e:
            print(f"Failed to generate admissible control visualization: {e}")

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
