import gymnasium
import numpy as np
import torch
from gymnasium import Env
from torch import nn

from maze_game import MazeGameEnv
from maze_game.sb3_tests.maze_cnn import MazeCNN

cnn = nn.Sequential(
    nn.Conv2d(26, 64, (3, 3), padding=1),
    nn.Tanh(),
    nn.Conv2d(64, 128, (3, 3), padding=1),
    nn.Tanh(),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.Tanh(),
    nn.Flatten(),
)

sts = nn.Sequential(
    nn.Conv1d(6, 12, 4),
    # nn.Flatten(),
)


def test_cnn(env: MazeGameEnv | Env):
    done = False
    obs, info = env.reset()
    field = torch.Tensor(obs['field'])
    stats = torch.Tensor(obs['stats'])
    # res = cnn(field)
    res = sts(stats)
    print(res)


if __name__ == "__main__":
    env = gymnasium.make('env_maze/MazeGame-v0', render_mode='human')
    test_cnn(env)
