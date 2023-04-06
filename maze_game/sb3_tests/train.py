from typing import Callable

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

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
        return (0.9 * progress_remaining + 0.1) * initial_value

    return func


def train(checkpoint=False):
    monitor_kwargs = dict(
        info_keywords=("is_success",)
    )
    env = make_vec_env("env_maze/MazeGame-v0", n_envs=16, monitor_kwargs=monitor_kwargs, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    model_kwargs = dict(
        learning_rate=linear_schedule(0.002),
        # learning_rate=0.001,
        n_epochs=4,
        batch_size=512,
        n_steps=128,
        ent_coef=0.01,
        vf_coef=0.25,
    )
    learn_kwargs = dict(
        total_timesteps=1_000_000
    )
    model = MaskablePPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log='storage',
                        **model_kwargs)
    if checkpoint:
        model.set_parameters("storage/ppo_MazeGame")
    print(model.policy)
    model.learn(**learn_kwargs)
    model.save("storage/ppo_MazeGame")


if __name__ == '__main__':
    train(checkpoint=False)
