"""Hooks for interoperability with imitation."""
import random
from typing import Sequence, Optional, cast

import numpy as np
from imitation.rewards import reward_nets
from imitation.data import types as imit_types, rollout as imit_rollout


from epic import types, samplers


def reward_net_to_fn(reward_net: reward_nets.RewardNet) -> types.RewardFunction:
    """Converts an imitation reward net to a reward function."""

    def reward_fn(state, action, next_state, done):
        return reward_net.forward(state, action, next_state, done).cpu().detach().numpy()

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
