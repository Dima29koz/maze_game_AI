import gymnasium
import numpy as np
from gymnasium import Env

from maze_game import MazeGameEnv


def run_random_with_mask(env: MazeGameEnv | Env):
    done = False
    obs = env.reset()
    while not done:
        mask = np.array(env.action_masks(), dtype=np.int8)
        action = env.action_space.sample(mask=mask)
        obs, reward, done, _, info = env.step(action)
        env.render()


if __name__ == "__main__":
    env = gymnasium.make('env_maze/MazeGame-v0', render_mode='human')
    run_random_with_mask(env)
