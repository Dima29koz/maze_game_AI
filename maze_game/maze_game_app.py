import gymnasium
from gymnasium import Env

from maze_game import MazeGameEnv


def run_random(env):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        env.render()


def run_random_with_mask(env: MazeGameEnv | Env):
    done = False
    obs = env.reset()
    while not done:
        mask = env.get_allowed_actions_mask()
        action = env.action_space.sample(mask=mask)
        obs, reward, done, _, info = env.step(action)
        env.render()


if __name__ == "__main__":
    env = gymnasium.make('env_maze/MazeGame-v0', render_mode='human')
    run_random(env)
    # run_random_with_mask(env)
