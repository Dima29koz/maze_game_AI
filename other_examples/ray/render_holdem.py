import argparse
import os

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from holdem import TorchMaskedActions

from pettingzoo.classic import leduc_holdem_v4

os.environ["SDL_VIDEODRIVER"] = "dummy"

parser = argparse.ArgumentParser(
    description="Render pretrained policy loaded from checkpoint"
)
parser.add_argument(
    "--checkpoint-path",
    help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
)

args = parser.parse_args()

_path = r"C:\Users\dima2\ray_results\DQN\DQN_leduc_holdem_v4_46fb4_00000_0_2023-04-14_11-42-23\checkpoint_000090"
checkpoint_path = os.path.expanduser(args.checkpoint_path or _path)


alg_name = "DQN"
ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
# function that outputs the environment you wish to register.


def env_creator():
    env = leduc_holdem_v4.env(render_mode='human')
    return env


env = env_creator()
env_name = "leduc_holdem_v4"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))


ray.init()
DQNAgent = Algorithm.from_checkpoint(checkpoint_path)

for _ in range(10):
    reward_sums = {a: 0 for a in env.possible_agents}
    i = 0
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        obs = observation["observation"]
        reward_sums[agent] += reward
        if termination or truncation:
            action = None
        else:
            print(DQNAgent.get_policy(agent))
            policy = DQNAgent.get_policy(agent)
            batch_obs = {
                "obs": {
                    "observation": np.expand_dims(observation["observation"], 0),
                    "action_mask": np.expand_dims(observation["action_mask"], 0),
                }
            }
            batched_action, state_out, info = policy.compute_actions_from_input_dict(
                batch_obs
            )
            single_action = batched_action[0]
            action = single_action

        env.step(action)
        i += 1
        env.render()

    print("rewards:")
    print(reward_sums)
