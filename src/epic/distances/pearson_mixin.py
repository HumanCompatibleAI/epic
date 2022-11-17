"""
Implements a mixin calculating the Pearson Distance between two reward
functions.
"""
from typing import Optional

import numpy as np

from epic import samplers, types


class PearsonMixin:
    """
    Mixin for the Pearson distance.
    """

    default_samples_cov: int
    default_samples_can: int
    coverage_sampler: samplers.BaseSampler[samplers.CoverageSample]

    def _distance(
        self,
        x_canonical: types.RewardFunction,
        y_canonical: types.RewardFunction,
        /,
        n_samples_cov: Optional[int],
        n_samples_can: Optional[int],
    ) -> float:
        """Returns the Pearson Distance between two canonicalized reward functions

        Args:
            x_canonical (types.RewardFunction): A canonicalized reward function
            y_canonical (types.RewardFunction): Another canonicalized reward function
            n_samples_cov (Optional[int]): Number of samples to draw from the coverage distribution of states and actions
            n_samples_can (Optional[int]): Number of samples to draw for canonicalization (unused here)

        Returns:
            float: The Pearson Distance between two reward functions, with special casing for one or both of the reward functions being constant. If both functions are constant, the distance is 0. If one function is constant, the distance is hard-coded to 0.5.
        """
        if isinstance(self.coverage_sampler, samplers.BaseDatasetSampler):
            action_cov_sample, state_cov_sample, next_state_cov_sample, done_cov_sample = self.coverage_sampler.sample(
                n_samples_cov,
            )
        else:
            action_cov_sample, state_cov_sample, next_state_cov_sample, done_cov_sample = self.coverage_sampler.sample(
                n_samples_cov or self.default_samples_cov,
            )

        x_samples = x_canonical(state_cov_sample, action_cov_sample, next_state_cov_sample, done_cov_sample)
        y_samples = y_canonical(state_cov_sample, action_cov_sample, next_state_cov_sample, done_cov_sample)

        # TODO: find a more permanent solution for this
        # handle cases with constant reward function
        if np.var(x_samples) < 1e-2 and np.var(y_samples) < 1e-2:
            return 0.0
        if np.var(x_samples) < 1e-2 or np.var(y_samples) < 1e-2:
            return 0.5
        else:
            return np.sqrt(1 - np.corrcoef(x_samples, y_samples)[0, 1])
