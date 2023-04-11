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
        features_dim: int = 256
    ) -> None:

        super().__init__(observation_space, features_dim)
        # We assume CxHxW observations (channels first)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (3, 3), padding=(1, 1)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(64, 128, (3, 3), padding=0),
            nn.Tanh(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten, features_dim),
        #     nn.ReLU()
        # )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # return self.linear(self.cnn(observations))
        return self.cnn(observations)


class StatsCNN(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 12
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxW observations (channels first)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, features_dim, 4),
            nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


class WallsCNN(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 256
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxW observations (channels first)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 32, (1, 3, 3), padding=(0, 1, 1)),
            nn.Tanh(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1)),
            nn.Tanh(),
            nn.MaxPool3d((2, 1, 1)),
            nn.Conv3d(64, 128, (1, 3, 3), padding=0),
            nn.Tanh(),
            nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


class TreasuresCNN(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 32
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxW observations (channels first)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, (3, 3), padding=(1, 1)),
            nn.Tanh(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, (3, 3), padding=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(16, 32, (3, 3), padding=0),
            nn.Tanh(),
            nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)
