from gymnasium.envs.registration import register

from .maze_game_env import MazeGameEnv

register(
    id="env_maze/MazeGame-v0",
    entry_point="maze_game.maze_game_env:MazeGameEnv",
)
