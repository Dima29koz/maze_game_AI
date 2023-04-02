from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from maze_game import MazeGameEnv
from maze_game.sb3_tests.features_extractor import CustomCombinedExtractor


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train():
    monitor_kwargs = dict(
        info_keywords=("is_success",)
    )
    env = make_vec_env("env_maze/MazeGame-v0", n_envs=16, monitor_kwargs=monitor_kwargs)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128),
    )
    model_kwargs = dict(
        learning_rate=linear_schedule(0.001),
        n_epochs=4,
        batch_size=512,
        n_steps=128,
        ent_coef=0.01,
    )
    learn_kwargs = dict(
        total_timesteps=4_000_000
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log='storage', **model_kwargs)
    print(model.policy)
    model.learn(**learn_kwargs)
    model.save("storage/ppo_MazeGame")


if __name__ == '__main__':
    train()
