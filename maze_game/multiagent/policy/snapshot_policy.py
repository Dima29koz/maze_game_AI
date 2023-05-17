import random
import torch

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2


class SnapshotPolicy(TorchPolicyV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view_requirements[SampleBatch.OBS].used_for_training = False

    def init_view_requirements(self):
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch columns
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.OBS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.PREV_ACTIONS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.REWARDS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.PREV_REWARDS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.UNROLL_ID]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.AGENT_INDEX]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.T]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.NEXT_OBS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.TRUNCATEDS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.TERMINATEDS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.ACTIONS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.EPS_ID]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.CUR_OBS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    def compute_actions_from_input_dict(self, input_dict, explore=None, timestep=None, **kwargs):
        with torch.no_grad():
            # Pass lazy (torch) tensor dict to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(False)
            return self._compute_action_helper(input_dict, [], None, explore, timestep)

    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def compute_gradients(self, postprocessed_batch):
        return [], {}

    def compute_log_likelihoods(
            self,
            actions,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            **kwargs,
    ):
        return np.array([random.random()] * len(obs_batch))

