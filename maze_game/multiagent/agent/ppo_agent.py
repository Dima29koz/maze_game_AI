import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from maze_game.multiagent.config import gen_policy
from maze_game.multiagent.maze_multi_agent_env import create_env
from maze_game.multiagent.models.action_masking import TorchActionMaskModel


def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return 'main'


class PPOAgent:
    def __init__(self, num_players: int, checkpoint_path: str):
        env_name = "maze_game_v2"
        register_env(env_name, lambda config: PettingZooEnv(create_env(num_players=num_players)))
        ModelCatalog.register_custom_model("pa_model", TorchActionMaskModel)

        ray.init(local_mode=True, num_cpus=0)
        config = (
            PPOConfig()
            .environment(env=env_name)
            .rollouts(
                num_rollout_workers=0,
            )
            .training(
                model={'custom_model': 'pa_model'}
            )
            .multi_agent(
                policies={f"main": gen_policy() for i in range(num_players)},
                policy_mapping_fn=new_policy_mapping_fn,
                policies_to_train=[],
            )
            .debugging(log_level="ERROR")
            .framework(framework="torch")
            .resources(num_gpus=1)
        )

        self.ppo_agent = PPO(config)
        self.ppo_agent.restore(checkpoint_path)

    def compute_action(self, observation):
        return self.ppo_agent.compute_single_action(observation, policy_id='main')
