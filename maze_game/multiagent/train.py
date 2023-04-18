import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from ray.tune.registry import register_env

from maze_game.multiagent.config import obs_space, act_space, env_name, num_players, policy_mapping_fn
from maze_game.multiagent.maze_multi_agent_env import create_env
from maze_game.multiagent.models.action_masking import TorchActionMaskModel


if __name__ == "__main__":
    ray.init()

    register_env(env_name, lambda conf: PettingZooEnv(create_env(num_players=num_players)))
    ModelCatalog.register_custom_model("pa_model", TorchActionMaskModel)

    def gen_policy():
        return PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=obs_space,
            action_space=act_space,
        )

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(
            num_rollout_workers=15,
            create_env_on_local_worker=False,
            rollout_fragment_length='auto',
        )
        .training(
            train_batch_size=512,
            lr=0.001,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=512,
            num_sgd_iter=4,
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

    tune.run(
        "PPO",
        name="maze_game_tune",
        stop={"timesteps_total": 1_000_000},
        checkpoint_freq=100,
        keep_checkpoints_num=10,
        config=config.to_dict(),
        sync_config=tune.SyncConfig(),
    )
