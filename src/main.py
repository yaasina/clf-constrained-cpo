from logger import Logger
from models.agent import Agent
from models.clf import CLFNetworkLightning
from models.dynamics import ControlAffineNetworkLightning

from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from copy import deepcopy
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
    epochs = 2500
    save_freq = 10
    algo = '{}_{}'.format(agent_name, algo_idx)
    save_name = '_'.join(env_name.split('-')[:-1])
    save_name = "result/{}_{}".format(save_name, algo)
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
    seed = algo_idx + random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    agent = Agent(env, device, args)
    clf = CLFNetworkLightning(env.observation_space.shape[0])
    dynamics = ControlAffineNetworkLightning(env.observation_space.shape[0], env.action_space.shape[0])

    # for wandb
    wandb.init(project='[torch] CPO')
    
    for epoch in range(epochs):
        trajectories = []
        ep_step = 0
        scores = []
        while ep_step < max_steps:
            state = env.reset()
            score = 0
            step = 0
            while True:
                ep_step += 1
                step += 1
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
                    break

            scores.append(score)

            
        states, actions, rewards, dones, fails, next_states = zip(*trajectories)
        batch = {
        "states": torch.as_tensor(states, dtype=torch.float32),
        "actions": torch.as_tensor(actions, dtype=torch.float32),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32),
        "dones": torch.as_tensor(dones, dtype=torch.float32),
        "fails": torch.as_tensor(fails, dtype=torch.float32),
        "next_states": torch.as_tensor(next_states, dtype=torch.float32)
        }

        dynamics_loss = dynamics.training_step(batch)
        clf_loss = clf._shared_step(batch)

        # calculate costs

        # add cost to trajectories

        v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajs=trajectories)
        score = np.mean(scores)
        log_data = {"score":score, "value loss":v_loss, "cost value loss":cost_v_loss, "objective":objective, "cost surrogate":cost_surrogate, "kl":kl, "entropy":entropy}
        print(log_data)
        wandb.log(log_data)
        if (epoch + 1)%save_freq == 0:
            agent.save()


def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPO')
    parser.add_argument('--test', action='store_true', help='For test.')
    parser.add_argument('--resume', type=int, default=0, help='type # of checkpoint.')
    parser.add_argument('--graph', action='store_true', help='For graph.')
    args = parser.parse_args()
    dict_args = vars(args)
    if args.test:
        test(args)
    else:
        train(args)
