from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from maze_game import MazeGameEnv
from maze_game.sb3_tests import config as conf


def train(checkpoint=False):
    env = make_vec_env(
        "env_maze/MazeGame-v0",
        n_envs=conf.n_envs, monitor_kwargs=conf.monitor_kwargs, vec_env_cls=SubprocVecEnv)

    if checkpoint:
        conf.run_id -= 1

    model = MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=conf.policy_kwargs, verbose=2, tensorboard_log=conf.root_path, **conf.model_kwargs)

    if checkpoint:
        max_step = 40_000  # todo
        model = model.load(f"{conf.root_path}/{conf.model_name}_{conf.run_id + 1}/save_{max_step}_steps", env)

    print(model.policy)

    model.learn(**conf.learn_kwargs, reset_num_timesteps=not checkpoint)
    model.save(f"{conf.root_path}/{conf.model_name}_{conf.run_id + 1}/model")


if __name__ == '__main__':
    train(checkpoint=False)
