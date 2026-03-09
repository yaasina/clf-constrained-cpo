import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional

# State convention throughout this file matches pendulum_dynamics.py: [sin θ, cos θ, θ̇]  (L5)

def get_pendulum_equilibrium() -> torch.Tensor:
    """
    Get the equilibrium state for the upright pendulum.

    Convention: [sin θ, cos θ, θ̇]
    Upright equilibrium: θ=0 → sin(0)=0, cos(0)=1, θ̇=0

    Returns:
        Tensor [0.0, 1.0, 0.0]
    """
    return torch.tensor([0.0, 1.0, 0.0])

def state_to_angle(state: torch.Tensor) -> torch.Tensor:
    """
    Convert pendulum state [sin(θ), cos(θ), θ̇] to [θ, θ̇].

    Args:
        state: Pendulum state tensor of shape [..., 3]

    Returns:
        Transformed state tensor of shape [..., 2] with [θ, θ̇]
    """
    sin_theta = state[..., 0]
    cos_theta = state[..., 1]
    theta_dot = state[..., 2]

    # atan2(sin, cos) correctly recovers θ in (-π, π]
    theta = torch.atan2(sin_theta, cos_theta)

    original_shape = list(state.shape)
    original_shape[-1] = 2

    return torch.stack([theta, theta_dot], dim=-1).reshape(original_shape)

def angle_to_state(angle_state: torch.Tensor) -> torch.Tensor:
    """
    Convert [θ, θ̇] to pendulum state [sin(θ), cos(θ), θ̇].

    Args:
        angle_state: Tensor of shape [..., 2] with [θ, θ̇]

    Returns:
        Pendulum state tensor of shape [..., 3] in [sin θ, cos θ, θ̇] convention
    """
    theta = angle_state[..., 0]
    theta_dot = angle_state[..., 1]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    original_shape = list(angle_state.shape)
    original_shape[-1] = 3

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
            grid_states: Tensor of shape [resolution*resolution, 3] in [sin θ, cos θ, θ̇]
            theta_grid: 2D grid of theta values
            theta_dot_grid: 2D grid of theta_dot values
    """
    theta = torch.linspace(theta_range[0], theta_range[1], resolution)
    theta_dot = torch.linspace(theta_dot_range[0], theta_dot_range[1], resolution)

    theta_grid, theta_dot_grid = torch.meshgrid(theta, theta_dot, indexing='ij')

    angle_states = torch.stack([theta_grid.flatten(), theta_dot_grid.flatten()], dim=-1)

    # angle_to_state now returns [sin θ, cos θ, θ̇] consistently
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
    grid_states, theta_grid, theta_dot_grid = create_pendulum_grid(
        resolution, theta_range, theta_dot_range
    )

    with torch.no_grad():
        values = value_function(grid_states).reshape(resolution, resolution)

    return values, theta_grid, theta_dot_grid
