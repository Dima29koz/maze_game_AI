import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from ray.tune.registry import register_env

from maze_game.multiagent.maze_multi_agent_env import create_env
from maze_game.multiagent.models.action_masking import TorchActionMaskModel

if __name__ == "__main__":
    ray.init()

    env_name = "maze_game_v2"
    num_players = 2

    register_env(env_name, lambda conf: PettingZooEnv(create_env(num_players=num_players)))
    ModelCatalog.register_custom_model("pa_model", TorchActionMaskModel)

    test_env = PettingZooEnv(create_env(num_players=num_players))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return PolicySpec(
            policy_class=None,  # infer automatically from Algorithm
            observation_space=obs_space,
            action_space=act_space,
        )

    policies = {f"policy_{i}": gen_policy() for i in range(num_players)}

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            model={'custom_model': 'pa_model'}
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id[-1],
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=1)
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )
