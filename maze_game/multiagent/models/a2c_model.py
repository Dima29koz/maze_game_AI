import torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from torch import nn
from gymnasium.spaces import Box
import numpy as np

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from maze_game.multiagent.models.complex_input import ComplexInputNetwork


class A2CNetwork(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and value heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(self, self.original_space, action_space, num_outputs, model_config, name)

        self.internal_model = ComplexInputNetwork(
            self.original_space,
            action_space,
            model_config,
            name + "_internal",
        )

        post_fc_stack_config = {
            "fcnet_hiddens": [512, 512, 512],
            "fcnet_activation": "tanh",
            "vf_share_layers": True,
        }
        self.policy_net = FullyConnectedNetwork(
            Box(-1, 1, shape=(self.internal_model.num_outputs,), dtype=np.float32),
            self.action_space,
            512,
            post_fc_stack_config,
            name="policy_net",
        )
        self.value_net = FullyConnectedNetwork(
            Box(-1, 1, shape=(self.internal_model.num_outputs,), dtype=np.float32),
            self.action_space,
            512,
            post_fc_stack_config,
            name="value_net",
        )

        # Actions and value heads.
        self._value_out = None

        # Action-distribution head.
        self.logits_layer = SlimFC(
            in_size=self.policy_net.num_outputs,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Create the value branch model.
        self.value_layer = SlimFC(
            in_size=self.value_net.num_outputs,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        out, _ = self.internal_model(SampleBatch({SampleBatch.OBS: input_dict['obs']}))

        pf_out, _ = self.policy_net(SampleBatch({SampleBatch.OBS: out}))
        vf_out, _ = self.value_net(SampleBatch({SampleBatch.OBS: out}))

        # Logits- and value branches.
        logits, values = self.logits_layer(pf_out), self.value_layer(vf_out)
        self._value_out = torch.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out
