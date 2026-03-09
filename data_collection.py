import numpy as np
import torch
import gymnasium as gym
from typing import List, Dict, Tuple, Any, Optional

def collect_trajectory(
    env: gym.Env, 
    length: int = 200, 
    random_seed: Optional[int] = None
) -> List[Dict[str, np.ndarray]]:
    """
    Collect a trajectory using a random policy.
    
    Args:
        env: Gymnasium environment
        length: Maximum length of trajectory
        random_seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing states, actions, and next_states
    """
    if random_seed is not None:
        env.action_space.seed(random_seed)
    
    state, _ = env.reset()
    trajectory = []
    
    for _ in range(length):
        # Sample random action
        action = env.action_space.sample()
        
        # Take a step
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Store transition
        trajectory.append({
            'state': state.copy(),
            'action': action.copy(),
            'next_state': next_state.copy()
        })
        
        # Update state
        state = next_state
        
        # Reset if done
        if terminated or truncated:
            state, _ = env.reset()
    
    return trajectory

def process_trajectory(
    trajectories: List[List[Dict[str, np.ndarray]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process trajectories into PyTorch tensors for training.
    
    Args:
        trajectories: List of trajectories, where each trajectory is a list of dictionaries
        
    Returns:
        Tuple containing:
            states: Tensor of states [batch_size, state_dim]
            actions: Tensor of actions [batch_size, action_dim]
            next_states: Tensor of next states [batch_size, state_dim]
    """
    states, actions, next_states = [], [], []
    
    for trajectory in trajectories:
        for transition in trajectory:
            states.append(transition['state'])
            actions.append(transition['action'])
            next_states.append(transition['next_state'])
    
    # Convert to numpy arrays
    states_np = np.array(states)
    actions_np = np.array(actions)
    next_states_np = np.array(next_states)
    
    # Convert to torch tensors
    states_tensor = torch.FloatTensor(states_np)
    actions_tensor = torch.FloatTensor(actions_np)
    next_states_tensor = torch.FloatTensor(next_states_np)
    
    return states_tensor, actions_tensor, next_states_tensor
