from models.agent import Agent
from models.dynamics import DynamicsEnsemble
from models.clf import CLFNetwork
from solvers.clf_qp_solver import CLFQPSolver

# from sklearn.utils import shuffle
# from collections import deque
# from scipy.stats import norm
# from copy import deepcopy
import numpy as np
import argparse
import pickle
import random
import torch
import wandb
import copy
import time
import gymnasium as gym

def train(main_args):
    algo_idx = 1
    agent_name = 'CPO'
    env_name = "Pendulum-v1"
    max_ep_len = 1000
    max_steps = 4000
    epochs = 250
    save_freq = 10
    seed = algo_idx + 55
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    if main_args["save_name"] is not None:
        save_name = main_args["save_name"]
    else:
        save_name = "result/{}_{}_{}".format(save_name, algo, seed)
    stop = False
    args = {
        'agent_name':agent_name,
        'save_name': save_name,
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
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')

    # for random seed
    np.random.seed(seed)
    random.seed(seed)


    equilibrium = torch.tensor([1.0, 0.0, 0.0])

    env = gym.make(env_name)
    agent = Agent(env, device, args)


    dynamics = DynamicsEnsemble(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_dim=64,
        ensemble_size=3, dt=0.05, learning_rate=1e-3, variance_tracker_size=max_steps
    ).to(device)

    clf = CLFNetwork(
        state_dim=env.observation_space.shape[0], hidden_dim=64, learning_rate=1e-3,
        loss={"alpha1": 1.0, "alpha2": 1.0, "alpha3": 0.1, "alpha4": 0.1},
        equilibrium=equilibrium, exp_const=1.0,
    ).to(device)

    qp = CLFQPSolver(action_dim=env.action_space.shape[0], action_limits=(-2.0, 2.0), exp_const=clf.exp_const).to(device)

    # Attach without registering as nn.Module submodules (keeps clf.parameters() CLF-only)
    object.__setattr__(clf, "dynamics_model", dynamics)
    object.__setattr__(clf, "qp_solver", qp)

    dyn_opt = dynamics.configure_optimizers()
    clf_opt = clf.configure_optimizers()

    # for wandb
    wandb.init(project='[torch] CPO')
    rew_arr = []
    step_arr = []
    cost_arr = []
    global_step = 0

    for epoch in range(epochs):
        trajectories = []
        ep_step = 0
        scores = []
        while ep_step < max_steps:
            state, _ = env.reset()
            score = 0
            step = 0
            while True:
                ep_step += 1
                step += 1
                global_step += 1

                state_tensor = torch.tensor(state, device=device, dtype=torch.float)
                action_tensor, clipped_action_tensor = agent.getAction(state_tensor, is_train=True)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                next_state, reward, terminated, truncated, info = env.step(clipped_action)

                done = terminated | truncated
                fail = terminated
                trajectories.append([state, action, reward, done, fail, next_state])

                state = next_state
                score += reward

                if done or step >= max_ep_len:
                    rew_arr.append(score)
                    step_arr.append(global_step)

                    break

            scores.append(score)

            
        states, actions, rewards, dones, fails, next_states = zip(*trajectories)
        batch = {
        "states": torch.as_tensor(states, device=device, dtype=torch.float32),
        "actions": torch.as_tensor(actions,device=device, dtype=torch.float32),
        "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float32),
        "dones": torch.as_tensor(dones, device=device, dtype=torch.float32),
        "fails": torch.as_tensor(fails, device=device, dtype=torch.float32),
        "next_states": torch.as_tensor(next_states, device=device, dtype=torch.float32)
        }

        # calculate costs
        clf.eval()
        dynamics.eval()
        costs = torch.relu(clf.compute_lie_derivative_with_action(batch["states"], batch["actions"], dynamics)).detach().cpu().numpy()
        cost_arr.append(costs)

        dynamics.train()
        dyn_opt.zero_grad()
        dynamics.compute_loss(batch["states"], batch["actions"], batch["next_states"]).backward()
        dyn_opt.step()

        # use dynamics in eval mode for training CLF
        clf.train()
        dynamics.eval()
        clf_opt.zero_grad()
        clf_losses = clf.compute_self_supervised_clf_loss(
            states=batch["states"], dynamics_model=dynamics, qp_solver=qp,
            next_states=batch["next_states"], dt=0.05,
            alpha1=clf.alpha1, alpha2=clf.alpha2, alpha3=clf.alpha3,
            alpha4=clf.alpha4, exp_const=clf.exp_const,
        )
        clf_total_loss = clf_losses["loss"]
        clf_total_loss.backward()
        clf_opt.step()

        # print(dynamics.dynamic_norm_c)
        # for i in range(uncert_update_freq):
        #     state_chunk = batch["states"][i*dynamics.variance_buffer_size:(i+1)*dynamics.variance_buffer_size]
        #     action_chunk = batch["actions"][i*dynamics.variance_buffer_size:(i+1)*dynamics.variance_buffer_size]
        #     dynamics.update_uncertainty(state_chunk, action_chunk)
        uncert = dynamics.get_normalized_uncertainty(batch["states"], batch["actions"])

        # add cost to trajectories
        trajectories = list(zip(states, actions, rewards, costs, dones, fails, next_states))

        v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajs=trajectories, uncert=uncert)
        score = np.mean(scores)

        smoothed_median = dynamics.variance_tracker.smoothed_median
        log_data = {"Episode Reward":score, "Total Steps": global_step, "Uncertainty": uncert.item()}
        for i in range(len(smoothed_median)):
            log_data[f"Variance Median Dim {i}"] = smoothed_median[i].item()
        log_data = {**log_data, **clf_losses}

        print(log_data)
        wandb.log(log_data)
        if (epoch + 1)%save_freq == 0:
            agent.save()
            clf.save_checkpoint(save_name + "/clf.pt")
            dynamics.save_checkpoint(save_name + "/dynamics.pt")
            np.savez(save_name + "_costs.npz", cost_arr=cost_arr)
            np.savez(save_name + "_rewards.npz", rew_arr=rew_arr)
            np.savez(save_name + "_steps.npz", step_arr=step_arr)
            if stop:
                break



def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPO')
    parser.add_argument('--test', action='store_true', help='For test.')
    parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--save_name', type=str, default=None, help='Name of base save directory')
    args = parser.parse_args()
    dict_args = vars(args)
    if args.test:
        test(args)
    else:
        train(dict_args)
