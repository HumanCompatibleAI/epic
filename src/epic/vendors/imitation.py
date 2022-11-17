"""Hooks for interoperability with imitation."""
import random
from typing import Sequence, Optional, cast

import numpy as np
import torch
from imitation.rewards import reward_nets
from imitation.data import types as imit_types, rollout as imit_rollout


from epic import types, samplers, utils


def reward_net_to_fn(reward_net: reward_nets.RewardNet, squeeze_output=False, device=None) -> types.RewardFunction:
    """Converts an imitation reward net to a reward function."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def reward_fn(state, action, next_state, done):

        state_tensor = utils.float_tensor_from_numpy(state, device)
        action_tensor = utils.float_tensor_from_numpy(action, device)
        next_state_tensor = utils.float_tensor_from_numpy(next_state, device)
        done_tensor = utils.float_tensor_from_numpy(done, device)

        if reward_net.use_state and state_tensor.ndim == 1:
            state_tensor.unsqueeze_(-1)
        if reward_net.use_action and action_tensor.ndim == 1:
            action_tensor.unsqueeze_(-1)
        if reward_net.use_next_state and next_state_tensor.ndim == 1:
            next_state_tensor.unsqueeze_(-1)

        reward_net.to(device)

        if squeeze_output:
            return utils.numpy_from_tensor(
                reward_net.forward(state_tensor, action_tensor, next_state_tensor, done_tensor).squeeze(-1)
            )
        return utils.numpy_from_tensor(reward_net.forward(state_tensor, action_tensor, next_state_tensor, done_tensor))

    return reward_fn


class TransitionSampler(samplers.DatasetSampler[samplers.CoverageSample, imit_types.Transitions]):
    def __init__(self, data: imit_types.Transitions, rng: np.random.Generator = None):
        super().__init__(data, rng)

    def sample(self, n_samples: Optional[int] = None):
        if n_samples and len(self.data) < n_samples:
            raise ValueError(
                f"n_samples ({n_samples}) must be less than " f"the number of data points ({len(self.data)})"
            )
        data = cast(imit_types.Transitions, random.sample(self.data, n_samples)) if n_samples else self.data
        return data.obs, data.acts, data.next_obs, data.dones


def sampler_from_trajs(trajectories: Sequence[imit_types.Trajectory], rng=None) -> TransitionSampler:
    transitions = imit_rollout.flatten_trajectories(trajectories)
    return TransitionSampler(transitions, rng=rng)
