from typing import Dict, List
import gymnasium as gym
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


def _create_cnn(obs_type: str, n_input_channels: int):
    match obs_type:
        case 'field':
            return nn.Sequential(
                nn.Conv2d(n_input_channels, 32, (3, 3), padding=(1, 1)),
                nn.Tanh(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
                nn.Tanh(),
                nn.Conv2d(64, 128, (3, 3), padding=0),
                nn.Tanh(),
                nn.Flatten(),
            ), 128
        case 'stats':
            return nn.Sequential(
                nn.Conv1d(n_input_channels, 16, 4),
                nn.Flatten(),
            ), 16
        case 'other_stats':
            # k x 6 x 4
            return nn.Sequential(
                nn.Conv2d(n_input_channels, 4 * n_input_channels, (1, 4), padding=0),
                nn.Tanh(),
                nn.Conv2d(4 * n_input_channels, 16 * n_input_channels, (6, 1), padding=0),
                nn.Tanh(),
                nn.Flatten(),
            ), 16 * n_input_channels
        case 'walls':
            return nn.Sequential(
                nn.Conv3d(n_input_channels, 32, (1, 3, 3), padding=(0, 1, 1)),
                nn.Tanh(),
                nn.MaxPool3d((2, 2, 2)),
                nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1)),
                nn.Tanh(),
                nn.MaxPool3d((2, 1, 1)),
                nn.Conv3d(64, 128, (1, 3, 3), padding=0),
                nn.Tanh(),
                nn.Flatten(),
            ), 128
        case 'treasures':
            return nn.Sequential(
                nn.Conv2d(n_input_channels, 8, (3, 3), padding=(1, 1)),
                nn.Tanh(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(8, 16, (3, 3), padding=(1, 1)),
                nn.Tanh(),
                nn.Conv2d(16, 32, (3, 3), padding=0),
                nn.Tanh(),
                nn.Flatten(),
            ), 32
        case _:
            raise Exception(f'unknown obs_type: {obs_type}')


class MazeCNN(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            model_config: ModelConfigDict,
            name: str,
    ):
        n_input_channels = obs_space.shape[0]
        conv_layers, num_outputs = _create_cnn(model_config.get('obs_type'), n_input_channels)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self._logits = None
        self._convs = conv_layers
        self._value_branch = SlimFC(
            num_outputs, 1, initializer=normc_initializer(0.01), activation_fn=None
        )

        self._features = None

    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        self._features = conv_out

        return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        features = self._features.squeeze(1)
        return self._value_branch(features)
