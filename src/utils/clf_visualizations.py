"""
Visualization utilities for Control Lyapunov Functions (CLF) and control analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gymnasium as gym

# Import utilities from pendulum_utils
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pendulum_utils import create_pendulum_grid, state_to_angle, angle_to_state, get_pendulum_equilibrium


def create_state_grid_from_env(
    env_name: str,
    resolution: int = 50,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create a grid of states based on the environment's observation space bounds.
    Specifically for pendulum, maps the 3D state space to a 2D representation.
    
    Args:
        env_name: Name of the gym environment
        resolution: Grid resolution per dimension
        device: Torch device to place tensors on
        
    Returns:
        Tuple containing:
            grid_states: Tensor of shape [resolution*resolution, state_dim]
            grid_info: Dictionary with additional grid information like theta_grid and theta_dot_grid
    """
    # Create environment to get observation space bounds
    env = gym.make(env_name)
    
    if 'Pendulum' in env_name:
        # For pendulum, we use a specific grid in theta-theta_dot space
        # Pendulum observation space is typically in range [(-1,-1,-8), (1,1,8)]
        # which represents [cos(θ), sin(θ), θ̇]
        
        # Get bounds from the environment
        low = env.observation_space.low
        high = env.observation_space.high
        
        # For pendulum specifically, we create a grid in [θ, θ̇] space
        theta_range = (-np.pi, np.pi)
        theta_dot_range = (low[2], high[2])  # Use actual bounds for angular velocity
        
        # Create the grid
        grid_states, theta_grid, theta_dot_grid = create_pendulum_grid(
            resolution=resolution,
            theta_range=theta_range,
            theta_dot_range=theta_dot_range
        )
        
        # Move to device
        grid_states = grid_states.to(device)
        
        grid_info = {
            'type': 'pendulum',
            'theta_grid': theta_grid,
            'theta_dot_grid': theta_dot_grid,
            'resolution': resolution,
            'theta_range': theta_range,
            'theta_dot_range': theta_dot_range
        }
    else:
        # For general environments with box observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            low = env.observation_space.low
            high = env.observation_space.high
            
            # Create grid of points
            grid_points = []
            
            if low.size == 1:
                # 1D state space
                grid_points = np.linspace(low[0], high[0], resolution)
                grid_states = torch.tensor(grid_points, dtype=torch.float32).unsqueeze(1).to(device)
                grid_info = {'type': 'general_1d', 'grid_points': grid_points}
            
            elif low.size == 2:
                # 2D state space
                x = np.linspace(low[0], high[0], resolution)
                y = np.linspace(low[1], high[1], resolution)
                X, Y = np.meshgrid(x, y)
                
                # Create tensor with all grid points
                points = np.stack([X.flatten(), Y.flatten()], axis=1)
                grid_states = torch.tensor(points, dtype=torch.float32).to(device)
                
                grid_info = {
                    'type': 'general_2d',
                    'x_grid': X,
                    'y_grid': Y,
                    'resolution': resolution
                }
            
            else:
                # Higher dimensional state space - create a grid on first two dimensions
                # and use middle values for other dimensions
                x = np.linspace(low[0], high[0], resolution)
                y = np.linspace(low[1], high[1], resolution)
                X, Y = np.meshgrid(x, y)
                
                # Create base points with middle values for dimensions > 2
                middle_values = (low[2:] + high[2:]) / 2
                base_point = np.concatenate([np.zeros(2), middle_values])
                
                # Create tensor with all grid points
                points = []
                for i in range(resolution):
                    for j in range(resolution):
                        point = base_point.copy()
                        point[0] = X[i, j]
                        point[1] = Y[i, j]
                        points.append(point)
                
                grid_states = torch.tensor(np.array(points), dtype=torch.float32).to(device)
                
                grid_info = {
                    'type': 'general_high_dim',
                    'x_grid': X,
                    'y_grid': Y,
                    'resolution': resolution,
                    'middle_values': middle_values
                }
        else:
            # For non-box observation spaces, we would need to implement custom logic
            raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")
    
    return grid_states, grid_info


def visualize_clf_value_grid(
    clf_model: torch.nn.Module,
    grid_states: torch.Tensor,
    grid_info: Dict[str, Any]
) -> go.Figure:
    """
    Visualize CLF values over a grid of states.
    
    Args:
        clf_model: The Control Lyapunov Function model
        grid_states: Grid of states to evaluate
        grid_info: Grid information dictionary
        
    Returns:
        Plotly figure object
    """
    # Compute CLF values on the grid
    with torch.no_grad():
        clf_values = clf_model.compute_clf(grid_states).cpu().numpy().flatten()
    
    # Create visualization based on grid type
    if grid_info['type'] == 'pendulum':
        # For pendulum, create a 2D contour plot in theta-theta_dot space
        resolution = grid_info['resolution']
        theta_grid = grid_info['theta_grid']
        theta_dot_grid = grid_info['theta_dot_grid']
        
        # Reshape values to match grid
        values_grid = clf_values.reshape(resolution, resolution)
        
        # Create figure
        fig = go.Figure(data=go.Contour(
            z=values_grid,
            x=theta_grid[:, 0].cpu().numpy(),  # theta values
            y=theta_dot_grid[0, :].cpu().numpy(),  # theta_dot values
            colorscale='Viridis',
            contours=dict(
                start=0,
                end=np.percentile(values_grid, 95),  # cut off at 95th percentile for better visualization
                size=(np.percentile(values_grid, 95) - 0) / 20,
                showlabels=True
            ),
            colorbar=dict(title='CLF Value'),
            name='CLF Value'
        ))
        
        # Add equilibrium point marker
        fig.add_trace(go.Scatter(
            x=[0],  # theta = 0 is the equilibrium
            y=[0],  # theta_dot = 0 is the equilibrium
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='star'
            ),
            name='Equilibrium Point'
        ))
        
        # Update layout
        fig.update_layout(
            title='CLF Value Over State Space',
            xaxis_title='θ (rad)',
            yaxis_title='θ̇ (rad/s)',
            width=600,
            height=500
        )
    
    else:
        # For general state spaces
        if grid_info['type'] == 'general_2d' or grid_info['type'] == 'general_high_dim':
            # 2D contour plot
            resolution = grid_info['resolution']
            x_grid = grid_info['x_grid']
            y_grid = grid_info['y_grid']
            
            # Reshape values to match grid
            values_grid = clf_values.reshape(resolution, resolution)
            
            # Create figure
            fig = go.Figure(data=go.Contour(
                z=values_grid,
                x=x_grid[0, :],
                y=y_grid[:, 0],
                colorscale='Viridis',
                colorbar=dict(title='CLF Value'),
                name='CLF Value'
            ))
            
            # Update layout
            fig.update_layout(
                title='CLF Value Over State Space',
                xaxis_title='State Dimension 1',
                yaxis_title='State Dimension 2',
                width=600,
                height=500
            )
        
        else:  # general_1d
            # 1D line plot
            grid_points = grid_info['grid_points']
            
            # Create figure
            fig = go.Figure(data=go.Scatter(
                x=grid_points,
                y=clf_values,
                mode='lines',
                name='CLF Value'
            ))
            
            # Update layout
            fig.update_layout(
                title='CLF Value Over State Space',
                xaxis_title='State',
                yaxis_title='CLF Value',
                width=600,
                height=400
            )
    
    return fig


def visualize_admissible_control_set(
    clf_model: torch.nn.Module,
    dynamics_model: torch.nn.Module,
    grid_states: torch.Tensor,
    grid_info: Dict[str, Any],
    action_bounds: Tuple[float, float],
    action_dim: int = 1,
    resolution: int = 100
) -> Tuple[go.Figure, torch.Tensor, torch.Tensor]:
    """
    Calculate and visualize the admissible control set where Lie derivative is non-positive.
    For each state, find the infimum of the action space where Lie derivative ≤ 0.
    
    Args:
        clf_model: The Control Lyapunov Function model
        dynamics_model: The system dynamics model
        grid_states: Grid of states to evaluate
        grid_info: Grid information dictionary
        action_bounds: Tuple of (min, max) action values
        action_dim: Dimension of the action space
        resolution: Resolution for action sampling
        
    Returns:
        Tuple containing:
            fig: Plotly figure with visualization
            infimum_actions: Tensor with infimum action values for each state
            lie_derivatives: Tensor with Lie derivative values for each state with its infimum action
    """
    # Only handle action_dim = 1 case for visualization
    if action_dim != 1:
        raise ValueError("Visualization of admissible control sets is only implemented for action_dim=1")
    
    # Get grid dimensions
    if grid_info['type'] == 'pendulum':
        grid_resolution = grid_info['resolution']
        theta_grid = grid_info['theta_grid'].cpu().numpy()
        theta_dot_grid = grid_info['theta_dot_grid'].cpu().numpy()
    else:
        raise ValueError("Admissible control set visualization is currently only implemented for pendulum")
    
    # Sample action space
    action_samples = torch.linspace(action_bounds[0], action_bounds[1], resolution).to(grid_states.device)
    
    # Initialize storage for infimum actions and their Lie derivatives
    infimum_actions = torch.full((len(grid_states),), float('inf'), device=grid_states.device)
    lie_derivatives_at_infimum = torch.zeros((len(grid_states),), device=grid_states.device)
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(grid_states), batch_size):
        batch_states = grid_states[i:i+batch_size]
        
        # Compute Lie derivatives for each state and all actions
        L_f_V, L_g_V = clf_model.lie_derivatives(batch_states, dynamics_model)
        
        # For each state in the batch
        for j in range(len(batch_states)):
            state_idx = i + j
            
            # Compute Lie derivative for all actions
            lie_derivatives = L_f_V[j] + L_g_V[j] * action_samples
            
            # Find actions that satisfy Lie derivative ≤ 0
            admissible_mask = lie_derivatives <= 0
            
            if torch.any(admissible_mask):
                # If there are admissible actions, find the infimum
                admissible_actions = action_samples[admissible_mask]
                
                # For stabilization, we want the smallest magnitude action that stabilizes
                # This approach finds the action closest to zero among admissible actions
                infimum_idx = torch.argmin(torch.abs(admissible_actions))
                infimum_action = admissible_actions[infimum_idx]
                
                # Store the infimum action and its Lie derivative
                infimum_actions[state_idx] = infimum_action
                lie_derivatives_at_infimum[state_idx] = lie_derivatives[admissible_mask][infimum_idx]
    
    # Reshape for visualization
    infimum_grid = infimum_actions.reshape(grid_resolution, grid_resolution).cpu().numpy()
    lie_derivatives_grid = lie_derivatives_at_infimum.reshape(grid_resolution, grid_resolution).cpu().numpy()
    
    # Replace points with no admissible action with NaN for visualization
    infimum_grid[infimum_grid == float('inf')] = np.nan
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Infimum Admissible Control", "Lie Derivative with Infimum Control"],
        specs=[[{"type": "contour"}, {"type": "contour"}]]
    )
    
    # Add infimum action contour
    fig.add_trace(
        go.Contour(
            z=infimum_grid,
            x=theta_grid[:, 0],  # theta values
            y=theta_dot_grid[0, :],  # theta_dot values
            colorscale='RdBu',
            colorbar=dict(title='Action Value', x=0.45),
            name='Infimum Admissible Control'
        ),
        row=1, col=1
    )
    
    # Add Lie derivative contour
    fig.add_trace(
        go.Contour(
            z=lie_derivatives_grid,
            x=theta_grid[:, 0],  # theta values
            y=theta_dot_grid[0, :],  # theta_dot values
            colorscale='Viridis',
            colorbar=dict(title='Lie Derivative', x=1.0),
            name='Lie Derivative'
        ),
        row=1, col=2
    )
    
    # Add equilibrium point marker to both subplots
    fig.add_trace(
        go.Scatter(
            x=[0],  # theta = 0 is the equilibrium
            y=[0],  # theta_dot = 0 is the equilibrium
            mode='markers',
            marker=dict(size=10, color='red', symbol='star'),
            name='Equilibrium Point',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[0],  # theta = 0 is the equilibrium
            y=[0],  # theta_dot = 0 is the equilibrium
            mode='markers',
            marker=dict(size=10, color='red', symbol='star'),
            name='Equilibrium Point'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Admissible Control Set Analysis',
        width=1200,
        height=600
    )
    
    fig.update_xaxes(title_text='θ (rad)', row=1, col=1)
    fig.update_yaxes(title_text='θ̇ (rad/s)', row=1, col=1)
    fig.update_xaxes(title_text='θ (rad)', row=1, col=2)
    fig.update_yaxes(title_text='θ̇ (rad/s)', row=1, col=2)
    
    return fig, infimum_actions, lie_derivatives_at_infimum