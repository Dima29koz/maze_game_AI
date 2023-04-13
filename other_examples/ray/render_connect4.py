import os
from time import sleep

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from pettingzoo.classic import connect_four_v3


path_ = r"C:\Users\dima2\ray_results\PPO\PPO_connect_four_v3_7c198_00000_0_2023-04-15_15-24-36\checkpoint_000190"
checkpoint_path = os.path.expanduser(path_)


def env_creator():
    return connect_four_v3.env(render_mode='human')


if __name__ == "__main__":

    env = env_creator()
    env_name = "connect_four_v3"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    ray.init(local_mode=True, num_cpus=0)
    PPOAgent = PPO.from_checkpoint(checkpoint_path)
    for _ in range(100):
        reward_sums = {a: 0 for a in env.possible_agents}
        i = 0
        env.reset()

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            reward_sums[agent] += reward
            if termination or truncation:
                action = None
            else:
                action = PPOAgent.compute_single_action(observation, policy_id=agent)
            env.step(action)
            env.render()
            i += 1
            sleep(0.6)

        print("rewards:")
        print(reward_sums)
