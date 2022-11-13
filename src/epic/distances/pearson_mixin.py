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
        if isinstance(self.coverage_sampler, samplers.BaseDatasetSampler):
            state_cov_sample, action_cov_sample, next_state_cov_sample, done_cov_sample = self.coverage_sampler.sample(
                n_samples_cov,
            )
        else:
            state_cov_sample, action_cov_sample, next_state_cov_sample, done_cov_sample = self.coverage_sampler.sample(
                n_samples_cov or self.default_samples_cov,
            )

        x_samples = x_canonical(state_cov_sample, action_cov_sample, next_state_cov_sample, done_cov_sample)
        y_samples = y_canonical(state_cov_sample, action_cov_sample, next_state_cov_sample, done_cov_sample)

        # handle cases with constant reward function
        if np.var(x_samples) < 1e-5 and np.var(y_samples) < 1e-5:
            return 0.0
        elif np.var(x_samples) < 1e-5 or np.var(y_samples) < 1e-5:
            return 0.5
        else:
            return np.sqrt(1 - np.corrcoef(x_samples, y_samples)[0, 1])
