import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# Pendulum dynamics
class PendulumDynamics(nn.Module):
    def __init__(self, mass=1.0, length=1.0, gravity=9.81, damping=0.01, equilibrium=[0, 1, 0]):
        super().__init__()
        self.m = mass
        self.L = length
        self.grav = gravity
        self.b = damping
        self.equilibrium = equilibrium
        
    def f(self, x):
        """
        Drift dynamics f(x)
        x: torch tensor of shape (batch_size, 3) containing [sin(theta), cos(theta), theta_dot]
        returns: torch tensor of shape (batch_size, 2)
        """
        # theta, theta_dot = x[:, 0], x[:, 1]
        sin_theta, cos_theta, theta_dot = x[:, 0], x[:, 1], x[:, 2]
        
        f1 = cos_theta
        f2 = -sin_theta
        f3 = self.grav/self.L * sin_theta - self.b/(self.m * self.L**2) * theta_dot
        
        return torch.stack([f1, f2, f3], dim=1)
    
    def g(self, x):
        """
        Control input matrix g(x)
        x: torch tensor of shape (batch_size, 2)
        returns: torch tensor of shape (batch_size, 2)
        """
        batch_size = x.shape[0]
        g1 = torch.zeros(batch_size, device=x.device)
        g2 = torch.zeros(batch_size, device=x.device)
        g3 = torch.full((batch_size,), 1/(self.m * self.L**2), device=x.device)
        
        return torch.stack([g1, g2, g3], dim=1)
    
    def forward(self, x, u):
        """
        x: torch tensor of shape (batch_size, 2) containing [theta, theta_dot]
        u: torch tensor of shape (batch_size, 1) containing control input
        returns: torch tensor of shape (batch_size, 2) containing [theta_dot, theta_ddot]
        """
        return self.f(x) + self.g(x) * u.squeeze()
    
    def equilibrium_point(self, x):
        # return torch.tensor(self.equilibrium)
        batch_size = x.shape[0]
        return torch.tensor(self.equilibrium).repeat(batch_size, 1) # (batch_size, 3)

# Lie derivative
class LieDerivative(nn.Module):
    def __init__(self, lyapunov, dynamics, exp_const=1):
        super().__init__()
        self.lyapunov = lyapunov
        self.dynamics = dynamics
        self.const = exp_const
        
    def forward(self, x, u):
        """
        Compute the Lie derivative of the Lyapunov function along system dynamics.
        
        Args:
            x: torch tensor of shape (batch_size, 3) containing [sin(theta), cos(theta), theta_dot]
               or single point of shape (3,)
            u: torch tensor of shape (batch_size,) or (batch_size, 1) or single value
               containing control input
               
        Returns:
            result: torch tensor of shape (batch_size,) or scalar containing Lie derivative values
        """
        # Add batch dimension if input is single point
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0) if torch.is_tensor(u) else torch.tensor([u], device=x.device)
        
        # Check dimensions
        if x.shape[1] != 3:
            raise ValueError(f"Expected x to have shape (batch_size, 3), got {x.shape}")
            
        V = self.lyapunov(x)  # (batch_size,)
        grad_V = self.lyapunov.gradient(x)  # (batch_size, 3)
        f = self.dynamics.f(x)  # (batch_size, 3)
        g = self.dynamics.g(x)  # (batch_size, 3)
        
        L_f_V = torch.sum(grad_V * f, dim=-1)  # (batch_size,)
        L_g_V = torch.sum(grad_V * g, dim=-1)  # (batch_size,)
        
        # Ensure u has correct shape (batch_size,)
        if torch.is_tensor(u):
            u = u.squeeze(-1) if u.dim() > x.dim()-1 else u  # (batch_size,)
        else:
            u = torch.full((x.shape[0],), u, device=x.device)
        
        result = L_f_V + L_g_V * u + self.const * V  # (batch_size,)
        
        # Remove batch dimension if input was single point
        if x.shape[0] == 1:
            result = result.squeeze(0)
            
        return result
    
    def evaluate_grid(self, grid_points, u=None):
        """
        Evaluate Lie derivative on provided grid points for plotting
        
        Args:
            grid_points: tensor of shape (n_points*n_points, 3) containing grid coordinates
                         [sin(theta), cos(theta), theta_dot]
            u: control input (float or callable that takes x as input and returns tensor of shape
               (n_points*n_points,) or (n_points*n_points, 1))
            
        Returns:
            lie_derivative: tensor of shape (n_points*n_points,) containing Lie derivative values
        """
        if grid_points.dim() != 2 or grid_points.shape[1] != 3:
            raise ValueError(f"Expected grid_points to have shape (n_points^2, 3), got {grid_points.shape}")
        
        x = grid_points  # shape (n_points*n_points, 3)
        
        if u is None:
            u = torch.zeros(x.shape[0], device=x.device)
        elif callable(u):
            u = u(x)
        else:
            u = torch.full((x.shape[0],), float(u), device=x.device)
            
        # Ensure u has correct shape
        if torch.is_tensor(u) and u.dim() > 1:
            u = u.squeeze(-1)
            
        lie_derivative = self.forward(x, u)
        
        return lie_derivative
    
# Pendulum dataset
class PendulumDataset(Dataset): 
    def __init__(self, n_samples=10000, theta_range=(-np.pi, np.pi), theta_dot_range=(-5, 5)):
        super().__init__()
        theta = torch.linspace(theta_range[0], theta_range[1], int(np.sqrt(n_samples)))
        theta_dot = torch.linspace(theta_dot_range[0], theta_dot_range[1], int(np.sqrt(n_samples)))
        
        # Create meshgrid
        theta_grid, theta_dot_grid = torch.meshgrid(theta, theta_dot, indexing='ij')
        
        # Convert theta to sin and cos
        sin_theta = torch.sin(theta_grid)
        cos_theta = torch.cos(theta_grid)
        
        # Stack and reshape to (n_samples, 3) where each row is [sin(theta), cos(theta), theta_dot]
        self.x = torch.stack([sin_theta, cos_theta, theta_dot_grid], dim=-1).reshape(-1, 3) # (n_samples, 3)
        # self.x_grid = torch.stack([theta_grid, theta_dot_grid], dim=-1)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]
    
    # # mesh grid of x for plotting
    # def x_grid(self):
    #     return self.x_grid