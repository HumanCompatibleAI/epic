"""Base sampling interface and implementations for canonicalization and coverage sampling."""


import abc
import random
from typing import Generic, Optional, Sequence, Tuple, TypeVar

import gym
import numpy as np
import numpy.typing as npt

T_co = TypeVar("T_co", covariant=True)


class BaseSampler(Generic[T_co], abc.ABC):
    @abc.abstractmethod
    def sample(self, n_samples: int, /) -> T_co:
        """Samples from the sampler.

        Returns: A numpy array of samples.
        """


class BaseDatasetSampler(BaseSampler[T_co], abc.ABC):
    @abc.abstractmethod
    def sample(self, n_samples: Optional[int] = None, /) -> T_co:
        """Sample from the dataset.

        Args:
            n_samples: The number of samples to draw. If ``None``, draw all samples.

        Returns:
            The sampled data.
        """


T_sized = TypeVar("T_sized", bound=Sequence)


class DatasetSampler(BaseDatasetSampler[T_co], Generic[T_co, T_sized]):
    """A sampler that samples from a preloaded dataset."""

    data: T_sized
    rng: np.random.Generator

    def __init__(self, data: T_sized, rng: Optional[np.random.Generator] = None):
        """Initializes the sampler.

        Args:
            data: The data to sample from.
            n_samples: The number of samples to take from the data. If None, all data is used
                (the array is returned with no modifications).
                If a value is specified, the first axis of the array is the dimension that the data
                is sampled from.

        Raises:
            ValueError: If the n_samples is larger than the first dimension of the data.
        """
        self.data = data
        self.rng = rng or np.random.default_rng()

    @abc.abstractmethod
    def sample(self, n_samples: Optional[int] = None):
        pass


class GymSamplerMixin:
    """Mixin class for samplers that sample from a gym space."""

    def __init__(self, space: gym.Space):
        self.space = space


class GymSpaceSampler(BaseSampler[npt.NDArray], GymSamplerMixin):
    """Samples from a gym space by using the gym space's sample method."""

    def sample(self, n_samples: int) -> npt.NDArray:
        return np.array([self.space.sample() for _ in range(n_samples)])


StateSample = Tuple[npt.NDArray[np.bool_], npt.NDArray]


class DummyGymStateSampler(BaseSampler[StateSample], GymSamplerMixin):
    """Samples from a gym space and returns a dummy done array."""

    def sample(self, n_samples: int) -> Tuple[npt.NDArray[np.bool_], npt.NDArray]:
        state_sample = np.array([self.space.sample() for _ in range(n_samples)])
        done_sample = np.zeros(n_samples, dtype=np.bool_)
        return done_sample, state_sample


CoverageSample = Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray[np.bool_]]


class ProductDistrCoverageSampler(BaseSampler[CoverageSample]):
    """Samples from a product distribution and returns the coverage."""

    def __init__(self, action_sampler: BaseSampler[npt.NDArray], state_sampler: BaseSampler[StateSample]):
        self.action_sampler = action_sampler
        self.state_sampler = state_sampler

    def sample(self, n_samples: int) -> CoverageSample:
        actions = self.action_sampler.sample(n_samples)
        _, states = self.state_sampler.sample(n_samples)
        dones, next_states = self.state_sampler.sample(n_samples)
        return actions, states, next_states, dones
