"""
Minimal standalone training pipeline.
gym rollout → update dynamics → update CLF → CLF cost + normalised uncertainty
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.dynamics import DynamicsEnsemble
from src.models.clf import CLFNetwork
from src.solvers.clf_qp_solver import CLFQPSolver

# ── config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cpu")
ENV_NAME   = "Pendulum-v1"
STATE_DIM  = 3
ACTION_DIM = 1
DT         = 0.05
EPOCHS     = 10
BATCH_SIZE = 128

equilibrium = torch.tensor([1.0, 0.0, 0.0])  # (cos θ, sin θ, θ̇) at θ=0

# ── models ────────────────────────────────────────────────────────────────────
dynamics = DynamicsEnsemble(
    state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=64,
    ensemble_size=3, dt=DT, learning_rate=1e-3,
).to(DEVICE)

clf = CLFNetwork(
    state_dim=STATE_DIM, hidden_dim=64, learning_rate=1e-3,
    loss={"alpha1": 1.0, "alpha2": 0.1, "alpha3": 1.0, "alpha4": 1.0},
    equilibrium=equilibrium, exp_const=1.0,
).to(DEVICE)

qp = CLFQPSolver(action_dim=ACTION_DIM, action_limits=(-2.0, 2.0), exp_const=clf.exp_const)

# Attach without registering as nn.Module submodules (keeps clf.parameters() CLF-only)
object.__setattr__(clf, "dynamics_model", dynamics)
object.__setattr__(clf, "qp_solver", qp)

dyn_opt = dynamics.configure_optimizers()
clf_opt = clf.configure_optimizers()

# ── rollout ───────────────────────────────────────────────────────────────────
env = gym.make(ENV_NAME)
obs_list, act_list, next_obs_list = [], [], []
obs, _ = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    next_obs, _, terminated, truncated, _ = env.step(action)
    obs_list.append(obs); act_list.append(action); next_obs_list.append(next_obs)
    obs = next_obs if not (terminated or truncated) else env.reset()[0]
env.close()

states      = torch.tensor(np.array(obs_list),      dtype=torch.float32, device=DEVICE)
actions     = torch.tensor(np.array(act_list),      dtype=torch.float32, device=DEVICE)
next_states = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=DEVICE)

# ── update dynamics ───────────────────────────────────────────────────────────
dynamics.train()
for s, a, ns in DataLoader(TensorDataset(states, actions, next_states), BATCH_SIZE, shuffle=True):
    dyn_opt.zero_grad()
    dynamics.compute_loss(s, a, ns).backward()
    dyn_opt.step()

# update adaptive normalisation constant from observed uncertainty
dynamics.eval()
with torch.no_grad():
    dynamics.update_dynamic_normalization_parameter(
        dynamics.compute_uncertainty(states, actions, use_mc_dropout=False)
    )

# ── update CLF ────────────────────────────────────────────────────────────────
clf.train(); dynamics.eval()
for s, ns in DataLoader(TensorDataset(states, next_states), BATCH_SIZE, shuffle=True):
    clf_opt.zero_grad()
    clf.compute_self_supervised_clf_loss(
        states=s, dynamics_model=dynamics, qp_solver=qp,
        next_states=ns, dt=DT,
        alpha1=clf.alpha1, alpha2=clf.alpha2, alpha3=clf.alpha3,
        alpha4=clf.alpha4, exp_const=clf.exp_const,
    )["loss"].backward()
    clf_opt.step()

# ── outputs: CLF cost + normalised uncertainty ────────────────────────────────
clf.eval(); dynamics.eval()
x = states[:32]   # example batch

# CLF cost: L_f_V + L_g_V·u* + μ·V  (should be ≤ 0 where CLF condition holds)
u_opt, r, failed = qp.solve_point(x[0:1], clf, dynamics)
clf_cost = clf.compute_lie_derivative_with_action(x[0:1], u_opt.unsqueeze(0), dynamics)

# normalised uncertainty ∈ [0, 1]
with torch.no_grad():
    raw_var  = dynamics.compute_uncertainty(x, actions[:32], use_mc_dropout=False)
    norm_var = dynamics.normalize_variance_dynamic(raw_var)

print(f"CLF cost  (L_f_V + L_g_V·u* + μ·V): {clf_cost.item():.4f}")
print(f"normalised uncertainty (mean):        {norm_var.mean().item():.4f}")
