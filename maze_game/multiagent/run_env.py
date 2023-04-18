import os

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from maze_game.multiagent.config import gen_policy, policy_mapping_fn
from maze_game.multiagent.maze_multi_agent_env import MAMazeGameEnv, create_env
from maze_game.multiagent.actions import action_to_action_space
from maze_game.multiagent.models.action_masking import TorchActionMaskModel


def manual_policy(env, agent=None, observation=None):
    act = None
    act_pl_abilities = env.unwrapped.game.get_allowed_abilities(env.unwrapped.game.get_current_player())
    while not act:
        act, state = env.unwrapped.gui.get_action(act_pl_abilities)
    return action_to_action_space(act)


def random_policy(env, agent, observation):
    return env.action_space(agent).sample(observation["action_mask"])


def ppo_policy(algo, agent, observation):
    return algo.compute_single_action(observation, policy_id=policy_mapping_fn(agent))


def run(num_resets=1):
    env_name = "maze_game_v2"
    num_players = 2

    env = create_env(render_mode="human", num_players=num_players)
    env.metadata['render_fps'] = 10

    register_env(env_name, lambda config: PettingZooEnv(create_env(num_players=num_players)))
    ModelCatalog.register_custom_model("pa_model", TorchActionMaskModel)

    path_ = r"C:\Users\dima2\ray_results\PPO\PPO_maze_game_v2_18043_00000_0_2023-04-18_19-00-09\checkpoint_001950"
    checkpoint_path = os.path.expanduser(path_)

    ray.init(local_mode=True, num_cpus=0)
    # ppo_agent = PPO.from_checkpoint(checkpoint_path)
    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .training(
            model={'custom_model': 'pa_model'}
        )
        .multi_agent(
            policies={f"policy_{i}": gen_policy() for i in range(num_players)},
            policy_mapping_fn=policy_mapping_fn,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=1)
    )

    ppo_agent = PPO(config)
    ppo_agent.restore(checkpoint_path)
    for _ in range(num_resets):
        env.reset()

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            print(info, f'reward={reward}')
            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                # action = random_policy(env, agent, observation)
                # action = manual_policy(env, agent, observation)
                action = ppo_policy(ppo_agent, agent, observation)

            env.step(action)

            env.render()
        env.reset()
    env.close()


if __name__ == "__main__":
    run(100)
