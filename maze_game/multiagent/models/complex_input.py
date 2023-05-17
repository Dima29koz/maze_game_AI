import torch
from torch import nn
from gymnasium.spaces import Box, Dict

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from maze_game.multiagent.models.maze_cnn import MazeCNN


class ComplexInputNetwork(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and value heads.
    """

    def __init__(self, obs_space: Dict, action_space, model_config, name):
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, self.original_space, action_space, 0, model_config, name
        )

        # Build the CNN(s) given obs_space's image components.
        self.cnns = nn.ModuleDict()
        self.observe_net = nn.ModuleDict()

        concat_size = 0
        for key, subspace in obs_space.spaces.items():
            assert isinstance(subspace, Box), f'subspace type should be `Box`, actual type is {type(subspace)}'
            self.cnns[key] = MazeCNN(
                subspace,
                action_space,
                model_config={"obs_type": key},
                name=f"cnn_{key}",
            )

            concat_size += self.cnns[key].num_outputs
            # idk if it`s needed
            # self.add_module("cnn_{}".format(key), self.cnns[key])

            # if key not in ["stats", "other_stats"]:
            #     self.observe_net[key] = nn.Flatten()
            #     if key == 'walls':
            #         sh = obs_space[key].shape
            #         concat_size += sh[0] * sh[2] * sh[3]
            #     else:
            #         concat_size += obs_space[key].sample().size

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        self.num_outputs = concat_size

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        outs = []
        for key, component in self.cnns.items():
            cnn_out, _ = component(SampleBatch({SampleBatch.OBS: input_dict['obs'][key]}))
            outs.append(cnn_out)
        # for key, component in self.observe_net.items():
        #     if key == 'walls':
        #         observe_net_out = component(input_dict['obs'][key][:, :, -1, :, :])
        #     else:
        #         observe_net_out = component(input_dict['obs'][key])
        #     outs.append(observe_net_out)
        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        return out, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out
