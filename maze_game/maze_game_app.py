import gymnasium
import torch_ac
from gymnasium import Env
from torch import nn

from maze_game import MazeGameEnv
from maze_game.utils import get_obss_preprocessor, preprocess_images, device

image_conv = nn.Sequential(
    nn.Conv2d(8, 16, (2, 2)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(16, 32, (2, 2)),
    nn.ReLU(),
    nn.Conv2d(32, 64, (2, 2)),
    nn.ReLU()
)


def test_conv(env):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        env.render()
        obs = torch_ac.DictList({
            "image": preprocess_images([state["field"]], device=device),
        })
        img = obs.image
        print(img)
        x = img.transpose(1, 3).transpose(2, 3)
        print(x)
        new_x = image_conv(x)
        print(new_x)
        new_x_s = new_x.reshape(new_x.shape[0], -1)
        print(new_x_s)
        break


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
    # test_conv(env)
