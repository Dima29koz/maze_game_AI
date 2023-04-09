import gymnasium
import numpy as np
import torch
from gymnasium import Env
from torch import nn

from maze_game import MazeGameEnv
from maze_game.sb3_tests.maze_cnn import MazeCNN

cnn = nn.Sequential(
    nn.Conv3d(26, 64, (1, 3, 3), padding=(0, 1, 1)),
    nn.Tanh(),
    nn.MaxPool3d((2, 2, 2)),
    nn.Conv3d(64, 128, (1, 3, 3), padding=(0, 1, 1)),
    nn.Tanh(),
    nn.MaxPool3d((2, 1, 1)),
    nn.Conv3d(128, 256, (1, 3, 3), padding=0),
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
    res = cnn(field)
    # res = sts(stats)
    model_parameters = filter(lambda p: p.requires_grad, cnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(res)


if __name__ == "__main__":
    env = gymnasium.make('env_maze/MazeGame-v0', render_mode='human')
    test_cnn(env)
