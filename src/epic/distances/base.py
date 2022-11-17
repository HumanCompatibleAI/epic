"""Base class for distance metrics."""


import abc
from typing import Optional

import numpy.typing as npt

from epic import samplers, types


class Distance(abc.ABC):
    """Compute the distance between two reward functions.

    Abstract base class to implement specific distance (pseudo-)metrics
    between reward functions. It consists of two steps:

    1. Canonicalization: Transform the reward function into a canonical form. This is so
        that equivalent reward functions are mapped to a single canonical form, for some
        notion of equivalence.
    2. Distance: Compute the distance between two reward functions in their
        canonical form.

    Users must implement the ``_distance`` method, which takes two reward functions
    in their canonical form and computes the distance between them, and the
    ``canonicalize`` method, which takes a reward function and returns its canonical
    form.
    """

    coverage_sampler: samplers.BaseSampler[samplers.CoverageSample]

    def __init__(
        self,
        discount_factor: float,
        coverage_sampler: samplers.BaseSampler[samplers.CoverageSample],
    ):
        self.coverage_sampler = coverage_sampler
        self.discount_factor = discount_factor

    @abc.abstractmethod
    def canonicalize(
        self,
        reward_function: types.RewardFunction,
        /,
        n_samples_can: Optional[int],
    ) -> types.RewardFunction:
        """Canonicalize a reward function.

        Args:
            reward_function: The reward function to canonicalize.
            n_samples_cov: The number of samples to draw from the coverage distribution.
            n_samples_can: The number of samples to draw for the canonicalization step
                for each sample of the coverage distribution. The total number of
                samples drawn is ``n_samples_cov * n_samples_can``.

        Returns:
            The canonicalized reward function.
        """

    @abc.abstractmethod
    def _distance(
        self,
        x_canonical: types.RewardFunction,
        y_canonical: types.RewardFunction,
        /,
        n_samples_cov: Optional[int],
        n_samples_can: Optional[int],
    ) -> float:
        """Subclass to implement the distance computation between two canonicalized
        reward functions.

        Args:
            x_canonical: The first canonicalized reward function.
            y_canonical: The second canonicalized reward function.

        Returns:
            The distance between the two reward functions.
        """
        raise NotImplementedError

    def distance(
        self,
        x: types.RewardFunction,
        y: types.RewardFunction,
        /,
        n_samples_cov: Optional[int] = None,
        n_samples_can: Optional[int] = None,
    ) -> float:
        """Compute the distance between two reward functions.

        Args:
            x: The first reward function.
            y: The second reward function.
            n_samples_cov: The number of samples to draw from the coverage distribution.
            n_samples_can: The number of samples to draw for the canonicalization step
                for each sample of the coverage distribution. The total number of
                samples drawn is ``n_samples_cov * n_samples_can``.

        Returns:
            The distance between the two reward functions.
        """
        x_canonical = self.canonicalize(x, n_samples_can)
        y_canonical = self.canonicalize(y, n_samples_can)
        return self._distance(x_canonical, y_canonical, n_samples_cov, n_samples_can)
