
import numpy as np
import random
import tree  # pip install dm_tree
from typing import (
    List,
    Optional,
    Union,
)

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType


class RandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    @override(Policy)
    def init_view_requirements(self):
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False
        vr = self.view_requirements[SampleBatch.OBS]
        vr.used_for_training = False
        vr.used_for_compute_actions = True
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

    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        **kwargs,
    ):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        obs_batch_size = len(tree.flatten(obs_batch)[0])
        obs_slice = np.int8(obs_batch[..., 0: 13])

        return (
            [self.action_space.sample(obs_slice[i]) for i in range(obs_batch_size)],
            [],
            {},
        )

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
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

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )
