import time

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch

from maze_game import utils
from maze_game.test.flat_obs import FlatObsWrapper
from maze_game.test.roll_out import RolloutBuffer
from maze_game import MazeGameEnv


def train(env, rollouts, policy, params, seed=123):
    # SETTING SEED: it is good practice to set seeds when running experiments to keep results comparable
    np.random.seed(seed)
    torch.manual_seed(seed)
    rewards, success_rate = [], []
    print("Training model with {} parameters...".format(policy.num_params))

    # Training Loop
    for j in range(params.num_updates):
        ## Initialization
        done = False
        prev_obs, info = env.reset()
        prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
        eps_reward = 0.
        start_time = time.time()

        ## Collect rollouts
        for step in range(rollouts.rollout_size):
            if done:
                rewards.append(eps_reward)
                if eps_reward > 0:
                    success_rate.append(1)
                else:
                    success_rate.append(0)
                # Reset Environment
                obs, info = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32)
                eps_reward = 0.

            else:
                obs = prev_obs

            action, log_prob = policy.act(obs)
            obs, reward, done, _, info = env.step(int(action))

            rollouts.insert(step, torch.tensor(done, dtype=torch.float32), action, log_prob,
                            torch.tensor(reward, dtype=torch.float32),
                            prev_obs)

            prev_obs = torch.tensor(obs, dtype=torch.float32)
            eps_reward += reward

        # Use the rollout buffer's function to compute the returns for all stored rollout steps. (requires just 1 line)
        rollouts.compute_returns(params.discount)

        rollout_done_time = time.time()

        # Call the policy's update function using the collected rollouts
        policy.update(rollouts)

        update_done_time = time.time()
        rollouts.reset()

    print(rewards, success_rate)
    return rewards, success_rate


def instantiate(params_in, nonwrapped_env=None):
    params = params_in

    if nonwrapped_env is None:
        nonwrapped_env = gym.make(params.env_name)
        nonwrapped_env = utils.make_env('env_maze/MazeGame-v0', 1)

    env = None
    # env = FlatObsWrapper(nonwrapped_env)
    env = FlattenObservation(nonwrapped_env)
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    rollouts = RolloutBuffer(params.rollout_size, obs_size)
    policy_class = params.policy_params.get('policy_class')
    pol = params.policy_params
    policy = policy_class(
        obs_size, num_actions,
        pol.get('hidden_dim'),
        pol.get('learning_rate'),
        pol.get('batch_size'),
        pol.get('policy_epochs'),
        pol.get('entropy_coef')
    )
    return env, rollouts, policy