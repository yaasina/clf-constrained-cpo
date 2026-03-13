"""
Module for QP solver using cvxpylayers for differentiable optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import plotly.graph_objects as go


class CLFQPSolver(nn.Module):
    """
    Quadratic Programming solver using cvxpylayers for Control Lyapunov Function.
    Solves the QP problem to find optimal control input that minimises the CLF derivative.

    QP:
        min  ||u||² + λ·r
        s.t. L_f_V + L_g_V @ u + μ·V ≤ r
             r ≥ 0
             action_lower ≤ u ≤ action_upper
    """

    def __init__(
        self,
        action_dim: int = None,
        action_limits: Tuple[float, float] = (-5.0, 5.0),
        lambda_param: float = 1.0,
        exp_const: float = 1.0,
        max_retries: int = 3,
        verbose: bool = False
    ) -> None:
        """
        Args:
            action_dim: Dimension of the action space (if None, determined lazily)
            action_limits: (lower, upper) bounds for actions
            lambda_param: Weight on the relaxation variable in the objective
            exp_const: Exponential constant for the CLF decrease condition
            max_retries: Maximum retries for failed optimisation
            verbose: Print verbose output during optimisation
        """
        super().__init__()

        self.action_dim = action_dim
        self.lambda_param = lambda_param
        self.action_lower = action_limits[0]
        self.action_upper = action_limits[1]
        self.const = exp_const
        self.max_retries = max_retries
        self.verbose = verbose

        # wandb.Run supplied externally for standalone evaluation (S5 pattern preserved)
        # Set qp_solver._eval_logger = wandb_run before calling solve_batch with log_results=True
        self._eval_logger = None  # wandb.Run or None

        # Tracking
        self.global_step: int = 0

        # QP setup flag — avoids ambiguity between _modules and __dict__ (S6)
        self._qp_ready: bool = False
        self.qp_layer = None  # replaced by CvxpyLayer in _setup_qp_problem

        if self.action_dim is not None:
            self._setup_qp_problem()

        self.reset_values()

    def _setup_qp_problem(self) -> None:
        """Set up the CVXPY QP and wrap it in a CvxpyLayer."""
        if self.action_dim is None:
            raise ValueError("action_dim must be set before setting up the QP problem")

        u = cp.Variable(self.action_dim)
        r = cp.Variable(1)
        L_f_V = cp.Parameter(1)
        L_g_V = cp.Parameter(self.action_dim)
        V = cp.Parameter(1)

        objective = cp.Minimize(cp.sum_squares(u) + self.lambda_param * r)
        constraints = [
            L_f_V + L_g_V @ u + self.const * V <= r,
            r >= 0,
            u <= self.action_upper,
            u >= self.action_lower,
        ]
        problem = cp.Problem(objective, constraints)
        self.qp_layer = CvxpyLayer(problem, parameters=[L_f_V, L_g_V, V], variables=[u, r], gp=False)
        self._qp_ready = True

    def reset_values(self) -> None:
        """Reset stored values from previous optimisations."""
        self.u_values: List[torch.Tensor] = []
        self.r_values: List[torch.Tensor] = []
        self.failed_states: List[torch.Tensor] = []
        self.solve_stats: Dict[str, int] = {
            'success_count': 0,
            'failure_count': 0,
            'retry_count': 0,
        }

    @property
    def _active_logger(self):
        """Return the W&B run for logging, or None if unavailable."""
        return self._eval_logger

    def solve_point(
        self,
        x: torch.Tensor,
        clf_net: nn.Module,
        dynamics_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Solve the QP for a single state point.

        Returns:
            u: optimal control
            r: relaxation variable (inf if failed)
            failed: True if optimisation failed
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        V = clf_net.compute_clf(x)
        L_f_V, L_g_V = clf_net.lie_derivatives(x, dynamics_model)

        V_detached = V.detach()
        L_f_V_detached = L_f_V.detach()
        L_g_V_detached = L_g_V.detach()

        # Lazy-init action_dim if not yet set
        if self.action_dim is None:
            if L_g_V_detached.dim() > 2:
                self.action_dim = L_g_V_detached.shape[2]
            else:
                self.action_dim = L_g_V_detached.shape[1]
            if self.action_dim is not None:
                self._setup_qp_problem()
                if self.verbose:
                    print(f"Automatically determined action_dim = {self.action_dim}")

        if not self._qp_ready:
            raise ValueError("QP solver not initialised. action_dim could not be determined.")

        for attempt in range(self.max_retries):
            try:
                if L_g_V_detached.dim() > 2:
                    L_g_V_shaped = L_g_V_detached.reshape(L_g_V_detached.shape[0], -1)
                else:
                    L_g_V_shaped = L_g_V_detached

                u, r = self.qp_layer(L_f_V_detached, L_g_V_shaped, V_detached)

                if torch.isnan(u).any() or torch.isinf(u).any() or torch.isnan(r).any() or torch.isinf(r).any():
                    if self.verbose:
                        print(f"Invalid solution (NaN/Inf) at attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        self.solve_stats['failure_count'] += 1
                        return torch.zeros(self.action_dim, device=x.device), torch.tensor([float('inf')], device=x.device), True
                    self.solve_stats['retry_count'] += 1
                    continue

                self.solve_stats['success_count'] += 1
                return u, r.squeeze(), False

            except Exception as e:
                if self.verbose:
                    print(f"Solver error at attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.solve_stats['failure_count'] += 1
                    return torch.zeros(self.action_dim, device=x.device), torch.tensor([float('inf')], device=x.device), True
                self.solve_stats['retry_count'] += 1

        return torch.zeros(self.action_dim, device=x.device), torch.tensor([float('inf')], device=x.device), True

    def solve_batch(
        self,
        dataset: torch.Tensor,
        clf_net: nn.Module,
        dynamics_model: nn.Module,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate the QP for every state in *dataset*.

        CLF values and Lie derivatives are computed in mini-batches for memory
        efficiency. The QP itself is solved point-by-point.

        Returns dict with: u_values, r_values, failed_states, failed_indices,
                           success_rate, stats
        """
        self.reset_values()
        device = dataset.device

        u_list = []
        r_list = []
        failed_indices = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]

            V_batch = clf_net.compute_clf(batch)
            L_f_V_batch, L_g_V_batch = clf_net.lie_derivatives(batch, dynamics_model)

            # Lazy-init action_dim (mirrors solve_point) — S6
            if self.action_dim is None:
                if L_g_V_batch.dim() > 2:
                    self.action_dim = L_g_V_batch.shape[2]
                else:
                    self.action_dim = L_g_V_batch.shape[1]
                if self.action_dim is not None:
                    self._setup_qp_problem()

            if not self._qp_ready:
                raise ValueError("QP solver not initialised. action_dim could not be determined.")

            for j in range(len(batch)):
                L_f_V = L_f_V_batch[j].reshape(1, 1)
                if L_g_V_batch.dim() > 2:
                    L_g_V = L_g_V_batch[j].reshape(1, self.action_dim)
                else:
                    L_g_V = L_g_V_batch[j].reshape(1, self.action_dim)
                V = V_batch[j].reshape(1, 1)

                try:
                    u, r = self.qp_layer(L_f_V, L_g_V, V)
                    failed = (
                        torch.isnan(u).any() or torch.isinf(u).any()
                        or torch.isnan(r).any() or torch.isinf(r).any()
                    )
                    if failed:
                        u = torch.zeros(self.action_dim, device=device)
                        r = torch.tensor([float('inf')], device=device)
                        failed_indices.append(i + j)
                        self.failed_states.append(batch[j].detach().cpu())
                        self.solve_stats['failure_count'] += 1
                    else:
                        self.solve_stats['success_count'] += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Solver failed: {str(e)}")
                        print(f"L_f_V shape: {L_f_V.shape}, L_g_V shape: {L_g_V.shape}, V shape: {V.shape}")
                    u = torch.zeros(self.action_dim, device=device)
                    r = torch.tensor([float('inf')], device=device)
                    failed_indices.append(i + j)
                    self.failed_states.append(batch[j].detach().cpu())
                    self.solve_stats['failure_count'] += 1

                u_list.append(u.detach())
                r_list.append(r.detach())

        u_values = torch.cat(u_list) if u_list else torch.tensor([], device=device)
        r_values = torch.cat(r_list) if r_list else torch.tensor([], device=device)

        total_points = len(dataset)
        success_rate = 100.0 * self.solve_stats['success_count'] / total_points if total_points > 0 else 0.0

        if self.verbose:
            print(f"QP Solver stats: {self.solve_stats}")
            print(f"Success rate: {success_rate:.2f}%")

        return {
            'u_values': u_values,
            'r_values': r_values,
            'failed_states': self.failed_states,
            'failed_indices': failed_indices,
            'success_rate': success_rate,
            'stats': self.solve_stats,
        }

    def get_control_policy(self, clf_net: nn.Module, dynamics_model: nn.Module) -> Callable:
        """Returns a callable policy u = policy(x)."""
        def control_policy(x: torch.Tensor) -> torch.Tensor:
            u, _, failed = self.solve_point(x, clf_net, dynamics_model)
            if failed:
                return torch.zeros(self.action_dim, device=x.device)
            return u
        return control_policy

    def compute_admissible_control_set(
        self,
        state: torch.Tensor,
        clf_net: nn.Module,
        dynamics_model: nn.Module,
        num_samples: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a sampling of the admissible control set where Lie derivative ≤ 0.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        L_f_V, L_g_V = clf_net.lie_derivatives(state, dynamics_model)

        if L_g_V.dim() > 2:
            L_g_V = L_g_V.reshape(L_g_V.shape[0], -1)

        if self.action_dim == 1:
            control_samples = torch.linspace(
                self.action_lower, self.action_upper, num_samples, device=state.device
            ).unsqueeze(1)
            lie_derivatives = L_f_V + L_g_V * control_samples
            admissible_mask = lie_derivatives <= 0
            admissible_controls = control_samples[admissible_mask.squeeze()]

            return admissible_controls, lie_derivatives.squeeze()

        elif self.action_dim == 2:
            n = int(np.sqrt(num_samples))
            u1_samples = torch.linspace(self.action_lower, self.action_upper, n, device=state.device)
            u2_samples = torch.linspace(self.action_lower, self.action_upper, n, device=state.device)
            U1, U2 = torch.meshgrid(u1_samples, u2_samples, indexing='ij')
            control_samples = torch.stack([U1.flatten(), U2.flatten()], dim=1)

            # Vectorised Lie derivative for all control samples (P2)
            L_g_V_reshaped = L_g_V if L_g_V.shape[1] == self.action_dim else L_g_V.reshape(1, self.action_dim)
            lie_derivatives = L_f_V.squeeze() + (control_samples * L_g_V_reshaped).sum(dim=-1)

            admissible_mask = lie_derivatives <= 0
            admissible_controls = control_samples[admissible_mask]

            return admissible_controls, lie_derivatives

        else:
            u, _, failed = self.solve_point(state, clf_net, dynamics_model)
            if failed:
                return torch.tensor([]), torch.tensor([])
            L_g_V_reshaped = L_g_V if L_g_V.shape[1] == self.action_dim else L_g_V.reshape(-1, self.action_dim)
            lie_derivative = L_f_V + torch.sum(L_g_V_reshaped * u)
            return u.unsqueeze(0), lie_derivative.unsqueeze(0)

    def _log_qp_results_wandb(
        self,
        states: torch.Tensor,
        u_values: torch.Tensor,
        r_values: torch.Tensor,
        failed_indices: List[int],
        clf_net: nn.Module,
        dynamics_model: nn.Module
    ) -> None:
        """Log QP solver results to W&B (C1 — enable_grad for autograd)."""
        with torch.enable_grad():
            if self.action_dim == 1:
                state_norms = torch.norm(states, dim=1).cpu().numpy()

                fig = go.Figure()
                successful_mask = np.ones(len(states), dtype=bool)
                successful_mask[failed_indices] = False

                fig.add_trace(
                    go.Scatter(
                        x=state_norms[successful_mask],
                        y=u_values.squeeze().cpu().numpy(),
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=r_values.squeeze().cpu().numpy(),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Relaxation Value r")
                        ),
                        name="Successful QP Solutions"
                    )
                )
                if failed_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=state_norms[failed_indices],
                            y=np.zeros(len(failed_indices)),
                            mode="markers",
                            marker=dict(size=8, color="red", symbol="x"),
                            name="Failed QP Solutions"
                        )
                    )
                fig.update_layout(
                    title="QP Control Solutions vs. State Norm",
                    xaxis_title="State Norm ||x||",
                    yaxis_title="Control Value u",
                    showlegend=True
                )
                self._active_logger.log({"qp_control_vs_norm": fig}, step=self.global_step)

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=r_values.squeeze().cpu().numpy(), nbinsx=30, marker_color="blue"))
                fig.update_layout(
                    title="Distribution of Relaxation Values",
                    xaxis_title="Relaxation Value r",
                    yaxis_title="Count"
                )
                self._active_logger.log({"qp_relaxation_histogram": fig}, step=self.global_step)

            # Lie derivative with QP control (P3 — seeded randperm)
            max_samples = min(100, len(states))
            sample_indices = torch.randperm(len(states), device=states.device)[:max_samples]
            sampled_states = states[sample_indices]
            sampled_controls = u_values[sample_indices]

            lie_derivatives = []
            clf_values = []
            for i in range(len(sampled_states)):
                state = sampled_states[i:i + 1]
                control = sampled_controls[i:i + 1]
                clf_values.append(clf_net.compute_clf(state).item())
                lie_derivatives.append(clf_net.compute_lie_derivative_with_action(state, control, dynamics_model).item())

            state_norms = torch.norm(sampled_states, dim=1).cpu().numpy()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=state_norms,
                    y=lie_derivatives,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=clf_values,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="CLF Value")
                    ),
                    name="Lie Derivative with QP Control"
                )
            )
            fig.add_hline(y=0, line=dict(color="red", dash="dash"), name="Zero")
            fig.update_layout(
                title="Lie Derivative with QP Control vs. State Norm",
                xaxis_title="State Norm ||x||",
                yaxis_title="Lie Derivative",
                showlegend=True
            )
            self._active_logger.log({"lie_derivative_with_qp_control": fig}, step=self.global_step)

# Backward-compatibility alias
CLFQPSolverLightning = CLFQPSolver
