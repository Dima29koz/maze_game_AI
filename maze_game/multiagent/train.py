import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from ray.tune.registry import register_env

from maze_game.multiagent.config import env_name, num_players, obs_space, act_space
from maze_game.multiagent.maze_multi_agent_env import create_env
from maze_game.multiagent.models.action_masking import TorchActionMaskModel
from maze_game.multiagent.random_policy import RandomPolicy
from maze_game.multiagent.self_playing import SelfPlayCallback


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    agent_idx = int(agent_id[-1])
    policy = "main" if episode.episode_id % 2 == agent_idx else "random"
    return policy


if __name__ == "__main__":
    ray.init(num_cpus=16, num_gpus=1)
    # ray.init(local_mode=True, num_cpus=0)

    register_env(env_name, lambda conf: PettingZooEnv(create_env(num_players=num_players)))
    ModelCatalog.register_custom_model("pa_model", TorchActionMaskModel)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .callbacks(SelfPlayCallback)
        .rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=24,
        )
        .training(
            use_critic=True,
            lr=0.001,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.01,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=512,
            num_sgd_iter=4,
            model={'custom_model': 'pa_model'}
        )
        .multi_agent(
            policies={
                'main': PolicySpec(
                    policy_class=None,
                    observation_space=obs_space,
                    action_space=act_space,
                ),
                "random": PolicySpec(
                    policy_class=RandomPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],
        )
        .debugging(log_level="WARN")
        .framework(framework="torch")
        .resources(num_gpus=1, num_cpus_for_local_worker=16)
        .checkpointing(checkpoint_trainable_policies_only=True)
    )

    tune.run(
        "PPO",
        name="maze_game_tune",
        # stop={"timesteps_total": 10_000_000},
        stop={"timesteps_total": 1_000_000},
        checkpoint_freq=200,
        keep_checkpoints_num=5,
        checkpoint_at_end=True,
        config=config.to_dict(),
    )
    ray.shutdown()
