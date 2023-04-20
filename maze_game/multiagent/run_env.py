import argparse
import os

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from maze_game.multiagent.config import policy_mapping_fn, policies, obs_space, act_space
from maze_game.multiagent.maze_multi_agent_env import MAMazeGameEnv, create_env
from maze_game.multiagent.actions import action_to_action_space
from maze_game.multiagent.models.action_masking import TorchActionMaskModel


parser = argparse.ArgumentParser()
parser.add_argument("--path", help="checkpoint path")


def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return 'main'


def manual_policy(env, agent=None, observation=None):
    if env.unwrapped.gui is None:
        env.render()
    act = None
    act_pl_abilities = env.unwrapped.game.get_allowed_abilities(env.unwrapped.game.get_current_player())
    while not act:
        act, state = env.unwrapped.gui.get_action(act_pl_abilities)
    return action_to_action_space(act)


def random_policy(env, agent, observation):
    return env.action_space(agent).sample(observation["action_mask"])


def ppo_policy(algo, agent, observation):
    policy_id = policy_mapping_fn(agent)
    # policy_id = new_policy_mapping_fn(agent, None, None)
    return algo.compute_single_action(observation, policy_id=policy_id)


def run(num_resets=1):
    env_name = "maze_game_v2"
    num_players = 2

    env = create_env(render_mode="human", num_players=num_players)
    env.metadata['render_fps'] = 10

    register_env(env_name, lambda config: PettingZooEnv(create_env(num_players=num_players)))
    ModelCatalog.register_custom_model("pa_model", TorchActionMaskModel)

    args = parser.parse_args()
    checkpoint_path = os.path.expanduser(args.path)

    ray.init(local_mode=True, num_cpus=0)
    config = (
        PPOConfig()
        .environment(env=env_name)
        .training(
            model={'custom_model': 'pa_model'}
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # policies={
            #     'main': PolicySpec(
            #         policy_class=None,
            #         observation_space=obs_space,
            #         action_space=act_space,
            #     ),
            # },
            # policy_mapping_fn=new_policy_mapping_fn,
            policies_to_train=[],
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=1)
    )

    ppo_agent = PPO(config)
    ppo_agent.restore(checkpoint_path)
    for i in range(num_resets):
        env.reset()
        print('reset number', i)
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
    env.close()


if __name__ == "__main__":
    run(100)
