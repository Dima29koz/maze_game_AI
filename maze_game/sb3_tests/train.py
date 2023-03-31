from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from maze_game import MazeGameEnv
from maze_game.sb3_tests.features_extractor import CustomCombinedExtractor


def train():
    env = make_vec_env("env_maze/MazeGame-v0", n_envs=16)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128),
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log='storage')
    model.learn(total_timesteps=100_000)
    model.save("storage/ppo_MazeGame")


if __name__ == '__main__':
    train()
