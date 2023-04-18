from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from maze_game.multiagent.maze_multi_agent_env import create_env

num_players = 2
env_name = "maze_game_v2"

test_env = PettingZooEnv(create_env(num_players=num_players))
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    return PolicySpec(
        policy_class=None,  # infer automatically from Algorithm
        observation_space=obs_space,
        action_space=act_space,
    )


def policy_mapping_fn(agent_id, *args, **kwargs):
    return f'policy_{agent_id[-1]}'
