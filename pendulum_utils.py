import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional

def get_pendulum_equilibrium() -> torch.Tensor:
    """
    Get the equilibrium state for the pendulum environment.

    The state convention used throughout this project is [sin(θ), cos(θ), θ̇],
    matching pendulum_dynamics.py. The upright equilibrium is θ=0, so
    sin(0)=0, cos(0)=1, θ̇=0 → [0, 1, 0].

    Returns:
        Tensor representing the equilibrium state [0, 1, 0]
    """
    return torch.tensor([0.0, 1.0, 0.0])

def state_to_angle(state: torch.Tensor) -> torch.Tensor:
    """
    Convert pendulum state [sin(θ), cos(θ), θ̇] to [θ, θ̇].

    State convention: index 0 = sin(θ), index 1 = cos(θ), index 2 = θ̇.
    This matches the ordering used in pendulum_dynamics.py.

    Args:
        state: Pendulum state tensor of shape [..., 3]

    Returns:
        Transformed state tensor of shape [..., 2] with [θ, θ̇]
    """
    sin_theta = state[..., 0]
    cos_theta = state[..., 1]
    theta_dot = state[..., 2]

    # Convert sin and cos to angle (in radians)
    # Use atan2 to get the correct quadrant
    theta = torch.atan2(sin_theta, cos_theta)

    # Reshape to match original dimensions but with 2 features
    original_shape = list(state.shape)
    original_shape[-1] = 2

    # Stack theta and theta_dot
    return torch.stack([theta, theta_dot], dim=-1).reshape(original_shape)

def angle_to_state(angle_state: torch.Tensor) -> torch.Tensor:
    """
    Convert [θ, θ̇] to pendulum state [sin(θ), cos(θ), θ̇].

    State convention: index 0 = sin(θ), index 1 = cos(θ), index 2 = θ̇.
    This matches the ordering used in pendulum_dynamics.py.

    Args:
        angle_state: Tensor of shape [..., 2] with [θ, θ̇]

    Returns:
        Pendulum state tensor of shape [..., 3]
    """
    theta = angle_state[..., 0]
    theta_dot = angle_state[..., 1]

    # Convert angle to sin and cos
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Reshape to match original dimensions but with 3 features
    original_shape = list(angle_state.shape)
    original_shape[-1] = 3

    # Stack sin, cos, and theta_dot (matching pendulum_dynamics.py convention)
    return torch.stack([sin_theta, cos_theta, theta_dot], dim=-1).reshape(original_shape)

def create_pendulum_grid(
    resolution: int = 50,
    theta_range: Tuple[float, float] = (-np.pi, np.pi),
    theta_dot_range: Tuple[float, float] = (-8.0, 8.0)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a grid of pendulum states for visualization.
    
    Args:
        resolution: Grid resolution for each dimension
        theta_range: Range of theta values (radians)
        theta_dot_range: Range of angular velocity values
        
    Returns:
        Tuple containing:
            grid_states: Tensor of shape [resolution*resolution, 3] containing pendulum states
            theta_grid: 2D grid of theta values
            theta_dot_grid: 2D grid of theta_dot values
    """
    # Create 1D grids
    theta = torch.linspace(theta_range[0], theta_range[1], resolution)
    theta_dot = torch.linspace(theta_dot_range[0], theta_dot_range[1], resolution)
    
    # Create 2D mesh grid
    theta_grid, theta_dot_grid = torch.meshgrid(theta, theta_dot, indexing='ij')
    
    # Stack to create angle_states: [resolution, resolution, 2]
    angle_states = torch.stack([theta_grid.flatten(), theta_dot_grid.flatten()], dim=-1)
    
    # Convert to pendulum states: [resolution*resolution, 3]
    grid_states = angle_to_state(angle_states)
    
    return grid_states, theta_grid, theta_dot_grid

def compute_values_on_grid(
    value_function: callable,
    resolution: int = 50,
    theta_range: Tuple[float, float] = (-np.pi, np.pi),
    theta_dot_range: Tuple[float, float] = (-8.0, 8.0)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute values on a grid of pendulum states.
    
    Args:
        value_function: Function that takes a batch of states and returns a value
        resolution: Grid resolution for each dimension
        theta_range: Range of theta values (radians)
        theta_dot_range: Range of angular velocity values
        
    Returns:
        Tuple containing:
            values: 2D grid of computed values
            theta_grid: 2D grid of theta values
            theta_dot_grid: 2D grid of theta_dot values
    """
    # Create grid
    grid_states, theta_grid, theta_dot_grid = create_pendulum_grid(
        resolution, theta_range, theta_dot_range
    )
    
    # Compute values
    with torch.no_grad():
        values = value_function(grid_states).reshape(resolution, resolution)
    
    return values, theta_grid, theta_dot_grid
