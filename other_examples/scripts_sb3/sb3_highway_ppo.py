import gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    path = 'storage/highway_ppo/model'
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="highway_ppo/")
        # Train the agent
        print("{}\n".format(model.policy))
        model.learn(total_timesteps=200_000)
        # Save the agent
        model.save(path=path)

    model = PPO.load(path=path)
    env = gym.make("highway-fast-v0")
    env.config['duration'] = 400
    while True:
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            action = action.item(0)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
        print(f'done={done}, truncated={truncated}')
