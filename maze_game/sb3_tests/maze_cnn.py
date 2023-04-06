import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn


class MazeCNN(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512
    ) -> None:

        super().__init__(observation_space, features_dim)
        # We assume CxHxW observations (channels first)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, (3, 3), padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.Tanh(),
            nn.Flatten(),
            # nn.Conv2d(n_input_channels, 64, (3, 3), padding=1),  # 64 7 7
            # nn.Sigmoid(),
            # nn.MaxPool2d((2, 2)),  # 64 3 3
            # nn.Conv2d(64, 128, (3, 3), padding=1),  # 128 3 3
            # nn.Sigmoid(),
            # nn.Conv2d(128, 256, (2, 2)),  # 64 3 3
            # nn.Sigmoid(),
            # nn.Conv2d(256, 512, (2, 2)),
            # nn.Sigmoid(),
            # nn.Flatten(),
            # nn.Conv2d(n_input_channels, 64, (3, 3), padding=1),  # 16 7 7
            # # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Conv2d(64, 128, (3, 3), padding=1),  # 32 7 7
            # # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.MaxPool2d((2, 2)),  # 32 3 3
            # nn.Conv2d(128, 256, (3, 3), padding=1),  # 64 3 3
            # # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.Conv2d(256, 512, (3, 3), padding=1),
            # # nn.Tanh(),
            # nn.Sigmoid(),
            # nn.MaxPool2d((2, 2)),  # 128 1 1
            # nn.Flatten(),
            # nn.Conv2d(n_input_channels, 16, (2, 2)),
            # nn.Tanh(),
            # nn.MaxPool2d((2, 2)),
            # nn.Conv2d(16, 32, (2, 2)),
            # nn.Tanh(),
            # nn.Conv2d(32, 64, (2, 2)),
            # nn.Tanh(),
            # nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
