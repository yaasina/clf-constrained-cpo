"""
Module for QP solver using cvxpylayers for differentiable optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import pytorch_lightning as pl
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CLFQPSolverLightning(pl.LightningModule):
    """
    Quadratic Programming solver using cvxpylayers for Control Lyapunov Function.
    Solves the QP problem to find optimal control input that minimizes the Lyapunov derivative.
    
    Implemented with PyTorch Lightning for better integration with the rest of the pipeline.
    """
    
    def __init__(
        self, 
        action_dim: int,
        action_limits: Tuple[float, float] = (-5.0, 5.0),
        lambda_param: float = 1.0, 
        exp_const: float = 1.0,
        max_retries: int = 3,
        verbose: bool = False
    ) -> None:
        """
        Initialize the QP optimizer.
        
        Args:
            action_dim: Dimension of the action space
            action_limits: Tuple of (lower, upper) bounds for actions
            lambda_param: Weight on the relaxation variable in the objective function
            exp_const: Exponential constant for the Lyapunov function decrease rate
            max_retries: Maximum number of retries for failed optimization
            verbose: Whether to print verbose output during optimization
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.action_dim = action_dim
        self.lambda_param = lambda_param
        self.action_lower = action_limits[0]
        self.action_upper = action_limits[1]
        self.const = exp_const
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Setup the optimization problem
        self._setup_qp_problem()
        
        # Initialize storage for results
        self.reset_values()
        
    def _setup_qp_problem(self) -> None:
        """Set up the QP problem with cvxpy variables and constraints"""
        # Define QP variables and parameters
        u = cp.Variable(self.action_dim) 
        r = cp.Variable(1)
        L_f_V = cp.Parameter(1)
        L_g_V = cp.Parameter(self.action_dim)
        V = cp.Parameter(1)
        
        # Define objective: minimize control effort and relaxation term
        objective = cp.Minimize(cp.sum_squares(u) + self.lambda_param * r)
        
        # Define constraints:
        # 1. Lie derivative of V must be less than r (relaxed CLF constraint)
        # 2. Relaxation variable must be non-negative
        # 3. Control input must be within limits
        constraints = [
            L_f_V + L_g_V @ u + self.const * V <= r,
            r >= 0,
            u <= self.action_upper,
            u >= self.action_lower
        ]
        
        # Create problem
        problem = cp.Problem(objective, constraints)
        
        # Create CvxpyLayer for differentiable optimization
        self.qp_layer = CvxpyLayer(
            problem, 
            parameters=[L_f_V, L_g_V, V], 
            variables=[u, r],
            gp=False  # Not a geometric program
        )

    def reset_values(self) -> None:
        """Reset stored values from previous optimizations"""
        self.u_values: List[torch.Tensor] = []
        self.r_values: List[torch.Tensor] = []
        self.failed_states: List[torch.Tensor] = []
        self.solve_stats: Dict[str, int] = {
            'success_count': 0,
            'failure_count': 0,
            'retry_count': 0
        }

    def solve_point(
        self, 
        x: torch.Tensor, 
        clf_net: nn.Module, 
        dynamics_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Solve the QP problem for a single state point.
        
        Args:
            x: State tensor of shape (state_dim,) or (1, state_dim)
            clf_net: Control Lyapunov Function neural network
            dynamics_model: System dynamics model
            
        Returns:
            u: Optimal control input as a tensor
            r: Relaxation variable value as a scalar tensor
            failed: Boolean indicating whether optimization failed
        """
        # Ensure x is 2D with batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Get CLF value and its derivatives
        V = clf_net.compute_clf(x)
        L_f_V, L_g_V = clf_net.lie_derivatives(x, dynamics_model)
        
        # Ensure we have the correct shapes and detach for QP solver
        V_detached = V.detach()
        L_f_V_detached = L_f_V.detach()
        L_g_V_detached = L_g_V.detach()
        
        # Attempt to solve with retries
        for attempt in range(self.max_retries):
            try:
                # Make sure L_g_V has the right shape for the QP solver
                # It should be [batch_size, action_dim], and we need the first (only) element
                if L_g_V_detached.dim() > 2:
                    L_g_V_shaped = L_g_V_detached.reshape(L_g_V_detached.shape[0], -1)
                else:
                    L_g_V_shaped = L_g_V_detached
                
                u, r = self.qp_layer(L_f_V_detached, L_g_V_shaped, V_detached)
                
                # Verify the solution is valid
                if torch.isnan(u).any() or torch.isinf(u).any() or torch.isnan(r).any() or torch.isinf(r).any():
                    if self.verbose:
                        print(f"Invalid solution (NaN/Inf) at attempt {attempt+1}")
                    if attempt == self.max_retries - 1:
                        # Last attempt, return default values
                        self.solve_stats['failure_count'] += 1
                        return torch.zeros(self.action_dim, device=x.device), torch.tensor([float('inf')], device=x.device), True
                    self.solve_stats['retry_count'] += 1
                    continue
                
                # Solution found
                self.solve_stats['success_count'] += 1
                return u, r.squeeze(), False
                
            except Exception as e:
                if self.verbose:
                    print(f"Solver error at attempt {attempt+1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    # Last attempt, return default values
                    self.solve_stats['failure_count'] += 1
                    return torch.zeros(self.action_dim, device=x.device), torch.tensor([float('inf')], device=x.device), True
                self.solve_stats['retry_count'] += 1
                
        # Should never reach here, but just in case
        return torch.zeros(self.action_dim, device=x.device), torch.tensor([float('inf')], device=x.device), True

    def solve_batch(
        self, 
        dataset: torch.Tensor, 
        clf_net: nn.Module,
        dynamics_model: nn.Module, 
        batch_size: int = 32,
        log_results: bool = False
    ) -> Dict[str, Any]:
        """
        Solve the QP problem for a batch of states, processing in smaller batches for efficiency.
        
        Args:
            dataset: Dataset of state points with shape (n_samples, state_dim)
            clf_net: Control Lyapunov Function neural network
            dynamics_model: System dynamics model
            batch_size: Size of mini-batches for processing
            log_results: Whether to log visualization results to Weights & Biases
            
        Returns:
            Dictionary containing:
                u_values: Tensor of optimal control inputs
                r_values: Tensor of relaxation variable values
                failed_states: List of state points where optimization failed
                success_rate: Percentage of successful optimizations
        """
        self.reset_values()
        device = dataset.device
        
        u_list = []
        r_list = []
        failed_indices = []
        
        # Process data in batches for better memory management
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Calculate V, L_f_V, and L_g_V for the batch
            V_batch = clf_net.compute_clf(batch)
            L_f_V_batch, L_g_V_batch = clf_net.lie_derivatives(batch, dynamics_model)
            
            # Solve for each point in the batch
            for j in range(len(batch)):
                L_f_V = L_f_V_batch[j].reshape(1, 1)
                
                # Make sure L_g_V has the right shape for the QP solver
                if L_g_V_batch.dim() > 2:
                    # For dynamics models that return g as [batch, state_dim, action_dim]
                    L_g_V = L_g_V_batch[j].reshape(1, self.action_dim)
                else:
                    # For models that already return the right shape
                    L_g_V = L_g_V_batch[j].reshape(1, self.action_dim)
                
                V = V_batch[j].reshape(1, 1)
                
                try:
                    u, r = self.qp_layer(L_f_V, L_g_V, V)
                    failed = torch.isnan(u).any() or torch.isinf(u).any() or torch.isnan(r).any() or torch.isinf(r).any()
                    
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
                        print(f"Solver failed with error: {str(e)}")
                        # Print shapes for debugging
                        print(f"L_f_V shape: {L_f_V.shape}, L_g_V shape: {L_g_V.shape}, V shape: {V.shape}")
                    u = torch.zeros(self.action_dim, device=device)
                    r = torch.tensor([float('inf')], device=device)
                    failed_indices.append(i + j)
                    self.failed_states.append(batch[j].detach().cpu())
                    self.solve_stats['failure_count'] += 1
                
                u_list.append(u.detach())
                r_list.append(r.detach())

        # Combine results 
        u_values = torch.cat(u_list) if u_list else torch.tensor([], device=device)
        r_values = torch.cat(r_list) if r_list else torch.tensor([], device=device)
        
        total_points = len(dataset)
        success_rate = 100.0 * self.solve_stats['success_count'] / total_points if total_points > 0 else 0.0
        
        if self.verbose:
            print(f"QP Solver stats: {self.solve_stats}")
            print(f"Success rate: {success_rate:.2f}%")
            print(f"Failed points: {len(failed_indices)}/{total_points} ({len(failed_indices)/total_points*100:.2f}%)")
        
        # Log results to W&B if requested
        if log_results and hasattr(self, 'logger') and isinstance(self.logger, pl.loggers.WandbLogger):
            self._log_qp_results_wandb(dataset, u_values, r_values, failed_indices, clf_net, dynamics_model)
        
        return {
            'u_values': u_values,
            'r_values': r_values,
            'failed_states': self.failed_states,
            'failed_indices': failed_indices,
            'success_rate': success_rate,
            'stats': self.solve_stats
        }
    
    def get_control_policy(self, clf_net: nn.Module, dynamics_model: nn.Module) -> Callable:
        """
        Returns a callable control policy function that computes the optimal
        control input for any given state using the QP solver.
        
        Args:
            clf_net: Control Lyapunov Function neural network
            dynamics_model: System dynamics model
            
        Returns:
            control_policy: Function that takes state x and returns control u
        """
        def control_policy(x: torch.Tensor) -> torch.Tensor:
            u, _, failed = self.solve_point(x, clf_net, dynamics_model)
            if failed:
                # Return zero control if optimization failed
                return torch.zeros(self.action_dim, device=x.device)
            return u
        
        return control_policy
    
    def compute_admissible_control_set(
        self, 
        state: torch.Tensor, 
        clf_net: nn.Module,
        dynamics_model: nn.Module,
        num_samples: int = 1000,
        log_results: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a sampling of the admissible control set where Lie derivative <= 0.
        
        Args:
            state: Single state tensor [state_dim]
            clf_net: Control Lyapunov Function neural network
            dynamics_model: System dynamics model
            num_samples: Number of control samples to evaluate
            log_results: Whether to log visualization results to Weights & Biases
            
        Returns:
            Tuple containing:
                admissible_controls: Tensor of admissible control values
                lie_derivatives: Lie derivative values for the sampled controls
        """
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Get CLF value and its derivatives
        L_f_V, L_g_V = clf_net.lie_derivatives(state, dynamics_model)
        
        # Make sure L_g_V has the right shape
        if L_g_V.dim() > 2:
            L_g_V = L_g_V.reshape(L_g_V.shape[0], -1)
        
        # For 1D action space, sample uniformly between limits
        if self.action_dim == 1:
            control_samples = torch.linspace(
                self.action_lower, 
                self.action_upper, 
                num_samples, 
                device=state.device
            ).unsqueeze(1)
            
            # Compute Lie derivative for each control
            lie_derivatives = L_f_V + L_g_V * control_samples
            
            # Find admissible controls
            admissible_mask = lie_derivatives <= 0
            admissible_controls = control_samples[admissible_mask.squeeze()]
            
            # Log visualization if requested
            if log_results and hasattr(self, 'logger') and isinstance(self.logger, pl.loggers.WandbLogger):
                self._log_admissible_controls_1d_wandb(
                    state, control_samples, lie_derivatives, admissible_controls, clf_net, dynamics_model
                )
            
            return admissible_controls, lie_derivatives.squeeze()
            
        # For multi-dimensional action spaces, sample from hypercube
        else:
            # We can only handle visualization for 1D and 2D action spaces currently
            if self.action_dim == 2:
                # Create a grid of control samples
                u1_samples = torch.linspace(self.action_lower, self.action_upper, int(np.sqrt(num_samples)), device=state.device)
                u2_samples = torch.linspace(self.action_lower, self.action_upper, int(np.sqrt(num_samples)), device=state.device)
                
                U1, U2 = torch.meshgrid(u1_samples, u2_samples, indexing='ij')
                control_samples = torch.stack([U1.flatten(), U2.flatten()], dim=1)
                
                # Compute Lie derivative for each control
                lie_derivatives = []
                for u in control_samples:
                    # Make sure L_g_V shape is compatible with control vector
                    if L_g_V.shape[1] != self.action_dim:
                        L_g_V_reshaped = L_g_V.reshape(-1, self.action_dim)
                    else:
                        L_g_V_reshaped = L_g_V
                    
                    lie_derivatives.append(L_f_V + torch.sum(L_g_V_reshaped * u))
                
                lie_derivatives = torch.tensor(lie_derivatives, device=state.device)
                
                # Find admissible controls
                admissible_mask = lie_derivatives <= 0
                admissible_controls = control_samples[admissible_mask]
                
                # Log visualization if requested
                if log_results and hasattr(self, 'logger') and isinstance(self.logger, pl.loggers.WandbLogger):
                    self._log_admissible_controls_2d_wandb(
                        state, control_samples, lie_derivatives, admissible_controls, 
                        u1_samples, u2_samples, U1, U2, clf_net, dynamics_model
                    )
                
                return admissible_controls, lie_derivatives
            
            # For higher dimensions, we'll simply return the optimal control
            else:
                u, _, failed = self.solve_point(state, clf_net, dynamics_model)
                if failed:
                    return torch.tensor([]), torch.tensor([])
                
                # Ensure L_g_V has the right shape
                if L_g_V.shape[1] != self.action_dim:
                    L_g_V_reshaped = L_g_V.reshape(-1, self.action_dim)
                else:
                    L_g_V_reshaped = L_g_V
                
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
        """
        Log QP solver results to Weights & Biases.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            u_values: Batch of control inputs [batch_size, action_dim]
            r_values: Batch of relaxation variables [batch_size, 1]
            failed_indices: List of indices where optimization failed
            clf_net: Control Lyapunov Function neural network
            dynamics_model: System dynamics model
        """
        with torch.no_grad():
            # Create scatter plot of control values vs. state norm
            if self.action_dim == 1:
                # 1D control, can visualize directly
                state_norms = torch.norm(states, dim=1).cpu().numpy()
                
                # Create figure
                fig = go.Figure()
                
                # Plot control values for successful points
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
                
                # Plot failed points
                if failed_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=state_norms[failed_indices],
                            y=np.zeros(len(failed_indices)),
                            mode="markers",
                            marker=dict(
                                size=8,
                                color="red",
                                symbol="x"
                            ),
                            name="Failed QP Solutions"
                        )
                    )
                
                fig.update_layout(
                    title="QP Control Solutions vs. State Norm",
                    xaxis_title="State Norm ||x||",
                    yaxis_title="Control Value u",
                    showlegend=True
                )
                
                # Log to wandb
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    wandb.log({"qp_control_vs_norm": fig})
                    
                # Create histogram of relaxation values
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=r_values.squeeze().cpu().numpy(),
                        nbinsx=30,
                        marker_color="blue"
                    )
                )
                
                fig.update_layout(
                    title="Distribution of Relaxation Values",
                    xaxis_title="Relaxation Value r",
                    yaxis_title="Count"
                )
                
                # Log to wandb
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    wandb.log({"qp_relaxation_histogram": fig})
            
            # Create visualization of resulting Lie derivatives with the computed control
            # Sample a subset of states for visualization
            max_samples = min(100, len(states))
            sample_indices = np.random.choice(len(states), max_samples, replace=False)
            
            sampled_states = states[sample_indices]
            sampled_controls = u_values[sample_indices]
            
            # Compute Lie derivatives with the computed control
            lie_derivatives = []
            clf_values = []
            
            for i in range(len(sampled_states)):
                state = sampled_states[i:i+1]
                control = sampled_controls[i:i+1]
                
                # Compute CLF value
                clf_value = clf_net.compute_clf(state)
                clf_values.append(clf_value.item())
                
                # Compute Lie derivative with the computed control
                lie_derivative = clf_net.compute_lie_derivative_with_action(state, control, dynamics_model)
                lie_derivatives.append(lie_derivative.item())
            
            # Create scatter plot of Lie derivatives with computed control
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
            
            # Add zero reference line
            fig.add_hline(y=0, line=dict(color="red", dash="dash"), name="Zero")
            
            fig.update_layout(
                title="Lie Derivative with QP Control vs. State Norm",
                xaxis_title="State Norm ||x||",
                yaxis_title="Lie Derivative",
                showlegend=True
            )
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                wandb.log({"lie_derivative_with_qp_control": fig})
    
    def _log_admissible_controls_1d_wandb(
        self,
        state: torch.Tensor,
        control_samples: torch.Tensor,
        lie_derivatives: torch.Tensor,
        admissible_controls: torch.Tensor,
        clf_net: nn.Module,
        dynamics_model: nn.Module
    ) -> None:
        """
        Log 1D admissible control visualization to Weights & Biases.
        
        Args:
            state: Single state tensor [1, state_dim]
            control_samples: Sampled control values [num_samples, 1]
            lie_derivatives: Lie derivative values for each control [num_samples, 1]
            admissible_controls: Subset of controls that are admissible
            clf_net: Control Lyapunov Function neural network
            dynamics_model: System dynamics model
        """
        with torch.no_grad():
            # Create figure
            fig = go.Figure()
            
            # Plot Lie derivative for each control sample
            fig.add_trace(
                go.Scatter(
                    x=control_samples.squeeze().cpu().numpy(),
                    y=lie_derivatives.squeeze().cpu().numpy(),
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="Lie Derivative"
                )
            )
            
            # Highlight admissible controls
            if len(admissible_controls) > 0:
                # Create admissible region by adding a colored area
                # First, determine the y-values corresponding to the admissible controls
                admissible_indices = []
                for u in admissible_controls:
                    idx = torch.argmin(torch.abs(control_samples - u)).item()
                    admissible_indices.append(idx)
                
                admissible_x = admissible_controls.squeeze().cpu().numpy()
                admissible_y = lie_derivatives.squeeze()[admissible_indices].cpu().numpy()
                
                # Add a colored area for admissible region
                fig.add_trace(
                    go.Scatter(
                        x=admissible_x,
                        y=admissible_y,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="green",
                            symbol="circle"
                        ),
                        name="Admissible Controls"
                    )
                )
                
                # Compute the optimal control from QP
                u_qp, _, _ = self.solve_point(state, clf_net, dynamics_model)
                
                # Find the closest control sample to the QP solution
                u_qp_idx = torch.argmin(torch.abs(control_samples - u_qp)).item()
                
                # Add QP solution to the plot
                fig.add_trace(
                    go.Scatter(
                        x=[u_qp.item()],
                        y=[lie_derivatives.squeeze()[u_qp_idx].item()],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color="red",
                            symbol="star"
                        ),
                        name="QP Solution"
                    )
                )
            
            # Add zero reference line
            fig.add_hline(y=0, line=dict(color="red", dash="dash"), name="Zero")
            
            # Update layout
            fig.update_layout(
                title=f"Lie Derivative vs. Control (State Norm: {torch.norm(state).item():.4f})",
                xaxis_title="Control u",
                yaxis_title="Lie Derivative",
                showlegend=True
            )
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                wandb.log({"admissible_controls_1d": fig})
    
    def _log_admissible_controls_2d_wandb(
        self,
        state: torch.Tensor,
        control_samples: torch.Tensor,
        lie_derivatives: torch.Tensor,
        admissible_controls: torch.Tensor,
        u1_samples: torch.Tensor,
        u2_samples: torch.Tensor,
        U1: torch.Tensor,
        U2: torch.Tensor,
        clf_net: nn.Module,
        dynamics_model: nn.Module
    ) -> None:
        """
        Log 2D admissible control visualization to Weights & Biases.
        
        Args:
            state: Single state tensor [1, state_dim]
            control_samples: Sampled control values [num_samples, 2]
            lie_derivatives: Lie derivative values for each control [num_samples]
            admissible_controls: Subset of controls that are admissible
            u1_samples, u2_samples: 1D control sample vectors for creating the grid
            U1, U2: 2D meshgrids of control samples
            clf_net: Control Lyapunov Function neural network
            dynamics_model: System dynamics model
        """
        with torch.no_grad():
            # Create contour plot of Lie derivative values
            grid_size = u1_samples.shape[0]
            lie_grid = lie_derivatives.reshape(grid_size, grid_size).cpu().numpy()
            
            # Create figure with subplots
            fig = make_subplots(
                rows=1, 
                cols=2,
                subplot_titles=[
                    "Lie Derivative Contour",
                    "Admissible Control Region"
                ],
                specs=[[{"type": "contour"}, {"type": "scatter"}]]
            )
            
            # Add contour plot
            fig.add_trace(
                go.Contour(
                    z=lie_grid,
                    x=u1_samples.cpu().numpy(),
                    y=u2_samples.cpu().numpy(),
                    colorscale="Viridis",
                    contours=dict(
                        start=-2,
                        end=2,
                        size=0.1,
                        showlabels=True
                    ),
                    colorbar=dict(
                        title="Lie Derivative",
                        x=0.46
                    ),
                    name="Lie Derivative"
                ),
                row=1, col=1
            )
            
            # Add zero contour line for better visibility
            fig.add_trace(
                go.Contour(
                    z=lie_grid,
                    x=u1_samples.cpu().numpy(),
                    y=u2_samples.cpu().numpy(),
                    contours=dict(
                        start=0,
                        end=0,
                        size=0,
                        showlabels=False,
                        coloring="lines"
                    ),
                    line=dict(color="red", width=2),
                    showscale=False,
                    name="Zero Lie Derivative"
                ),
                row=1, col=1
            )
            
            # Plot admissible region (scatter plot)
            if len(admissible_controls) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=admissible_controls[:, 0].cpu().numpy(),
                        y=admissible_controls[:, 1].cpu().numpy(),
                        mode="markers",
                        marker=dict(
                            size=5,
                            color="green",
                            opacity=0.5
                        ),
                        name="Admissible Controls"
                    ),
                    row=1, col=2
                )
                
                # Compute the optimal control from QP
                u_qp, _, _ = self.solve_point(state, clf_net, dynamics_model)
                
                # Add QP solution to the plot
                fig.add_trace(
                    go.Scatter(
                        x=[u_qp[0].item()],
                        y=[u_qp[1].item()],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color="red",
                            symbol="star"
                        ),
                        name="QP Solution"
                    ),
                    row=1, col=2
                )
                
                # Add action limits to the plot
                limit_x = np.array([self.action_lower, self.action_upper, self.action_upper, self.action_lower, self.action_lower])
                limit_y = np.array([self.action_lower, self.action_lower, self.action_upper, self.action_upper, self.action_lower])
                
                fig.add_trace(
                    go.Scatter(
                        x=limit_x,
                        y=limit_y,
                        mode="lines",
                        line=dict(color="black", dash="dash"),
                        name="Action Limits"
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"Admissible Control Region (State Norm: {torch.norm(state).item():.4f})",
                height=500,
                width=1000
            )
            
            fig.update_xaxes(title_text="Control u1", row=1, col=1)
            fig.update_yaxes(title_text="Control u2", row=1, col=1)
            fig.update_xaxes(title_text="Control u1", row=1, col=2)
            fig.update_yaxes(title_text="Control u2", row=1, col=2)
            
            # Log to wandb
            if isinstance(self.logger, pl.loggers.WandbLogger):
                wandb.log({"admissible_controls_2d": fig})