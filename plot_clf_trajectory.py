import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os

from src.models.clf import CLFNetwork
from src.models.dynamics import DynamicsEnsemble
from src.solvers.clf_qp_solver import CLFQPSolver
from src.models.agent import Agent


def plot_clf_trajectory():
    """
    Plots the learned CLF, a sample trajectory, and the running cost for a trained model.
    """
    # --- 1. Load Models ---
    device = torch.device("cpu")
    
    # Correctly locate model files relative to the project root
    clf_model_path = "result/Pendulum_CPO_1_19_clf.pt"
    dynamics_model_path = "result/Pendulum_CPO_1_19_dynamics.pt"
    agent_path = "result/Pendulum_CPO_1_19"

    # Check if model files exist
    if not os.path.exists(clf_model_path) or not os.path.exists(dynamics_model_path):
        print(f"Error: Model files not found.")
        print(f"Searched for: {os.path.abspath(clf_model_path)}")
        print(f"And: {os.path.abspath(dynamics_model_path)}")
        return

    clf = CLFNetwork.load_checkpoint(clf_model_path, map_location=device)
    dynamics = DynamicsEnsemble.load_checkpoint(dynamics_model_path, map_location=device)
    
    # Instantiate QP solver and attach models to CLF
    qp = CLFQPSolver(action_dim=1, action_limits=(-2.0, 2.0), exp_const=clf.exp_const)
    object.__setattr__(clf, "dynamics_model", dynamics)
    object.__setattr__(clf, "qp_solver", qp)
    
    clf.eval()
    dynamics.eval()

    # --- 2. Generate Sample Trajectory ---
    env = gym.make("Pendulum-v1")
    
    agent_name = 'CPO'
    args = {
        'agent_name':agent_name,
        'save_name': agent_path,
        'discount_factor':0.99,
        'hidden1':512,
        'hidden2':512,
        'v_lr':2e-4,
        'cost_v_lr':2e-4,
        'value_epochs':200,
        'batch_size':10000,
        'num_conjugate':10,
        'max_decay_num':10,
        'line_decay':0.8,
        'max_kl':0.001,
        'damping_coeff':0.01,
        'gae_coeff':0.97,
        'cost_d':25.0/1000.0,
    }
    agent = Agent(env, device, args)
    state, _ = env.reset(seed=20)
    trajectory_states = [state]
    trajectory_actions = []
    for _ in range(200):
        action, clipped_action = agent.getAction(torch.from_numpy(state).float().to(device), is_train=False)
        trajectory_actions.append(clipped_action.detach().cpu().numpy())
        state, _, _, _, _ = env.step(clipped_action.detach().cpu().numpy())
        trajectory_states.append(state)
    
    trajectory_states = np.array(trajectory_states)
    trajectory_actions = np.array(trajectory_actions)

    # --- 3. Calculate Running Cost (Lie Derivative) ---
    with torch.no_grad():
        states_tensor = torch.from_numpy(trajectory_states[:-1]).float().to(device)
        actions_tensor = torch.from_numpy(trajectory_actions).float().to(device)
        running_costs = clf.compute_lie_derivative_with_action(states_tensor, actions_tensor, dynamics)
        running_costs = running_costs.cpu().numpy()

    # --- 4. Create CLF Plot Data ---
    theta_range = np.linspace(-np.pi, np.pi, 100)
    theta_dot_range = np.linspace(-8, 8, 100)
    theta_grid, theta_dot_grid = np.meshgrid(theta_range, theta_dot_range)

    cos_theta_grid = np.cos(theta_grid)
    sin_theta_grid = np.sin(theta_grid)
    
    states_to_evaluate = np.stack([cos_theta_grid, sin_theta_grid, theta_dot_grid], axis=-1)
    states_to_evaluate = states_to_evaluate.reshape(-1, 3)
    
    with torch.no_grad():
        states_tensor = torch.from_numpy(states_to_evaluate).float().to(device)
        clf_values = clf.compute_clf(states_tensor)
        clf_values = clf_values.cpu().numpy().reshape(100, 100)

    # --- 5. Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
    
    # Subplot 1: CLF and Trajectory
    contour = ax1.contourf(theta_grid, theta_dot_grid, clf_values, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax1, label='CLF Value V(x)')
    
    traj_theta = np.arctan2(trajectory_states[:, 1], trajectory_states[:, 0])
    traj_theta_dot = trajectory_states[:, 2]
    ax1.plot(traj_theta, traj_theta_dot, 'r.-', label='Sample Trajectory', alpha=0.6)
    
    # Mark start and end points
    ax1.plot(traj_theta[0], traj_theta_dot[0], 'go', markersize=12, label='Start')
    ax1.plot(traj_theta[-1], traj_theta_dot[-1], 'bo', markersize=12, label='End')
    ax1.plot([0], [0], 'r*', markersize=15, label='Equilibrium')
    
    ax1.set_title('Learned CLF with Sample Trajectory (Seed 20)')
    ax1.set_xlabel('Theta (rad)')
    ax1.set_ylabel('Theta_dot (rad/s)')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Running Cost
    ax2.plot(running_costs)
    ax2.set_title('Running Cost (Lie Derivative) along Trajectory')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cost (Lie Derivative)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('clf_trajectory_and_cost_seed20.png')
    print("Plot saved to clf_trajectory_and_cost_seed20.png")

if __name__ == '__main__':
    plot_clf_trajectory()
