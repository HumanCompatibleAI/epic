"""Implements the EPIC distance (https://arxiv.org/abs/2006.13900)."""

from typing import Optional, TypeVar

import numpy as np
import numpy.typing as npt

from epic import samplers, types, utils
from epic.distances import base, pearson_mixin

T_co = TypeVar("T_co", covariant=True)


class EPIC(pearson_mixin.PearsonMixin, base.Distance):
    default_samples_cov = 500
    default_samples_can = 500

    def __init__(
        self,
        discount_factor: float,
        state_sampler: samplers.BaseSampler[samplers.StateSample],
        action_sampler: samplers.BaseSampler[npt.NDArray],
        coverage_sampler: Optional[samplers.BaseSampler[samplers.CoverageSample]] = None,
    ):
        """Initialize the EPIC distance.

        Args:
            state_sampler: The sampler for the state distribution.
            action_sampler: The sampler for the action distribution.
            coverage_sampler: The sampler for the coverage distribution. If not given,
                a default sampler is constructed as drawing from the product
                distribution induced by the distributions of state and action.
            discount_factor: The discount factor.
        """
        coverage_sampler = coverage_sampler or samplers.ProductDistrCoverageSampler(action_sampler, state_sampler)
        self.action_sampler = action_sampler
        self.state_sampler = state_sampler
        super().__init__(
            discount_factor,
            coverage_sampler,
        )

    def canonicalize(
        self,
        reward_function: types.RewardFunction,
        /,
        n_samples_can: Optional[int],
    ) -> types.RewardFunction:
        """Canonicalize a reward function.

        Applies the canonically shaped reward transformation defined in 4.1 of
        https://arxiv.org/pdf/2006.13900.pdf.

        Args:
            reward_function: The reward function to canonicalize.
            n_samples_cov: The number of samples to draw from the coverage distribution.
            n_samples_can: The number of samples to draw for the canonicalization step
                for each sample of the coverage distribution. The total number of
                samples drawn is ``n_samples_cov * n_samples_can``.
        """
        rew_fn = utils.multidim_rew_fn(reward_function)
        n_samples_can = n_samples_can or self.default_samples_can
        assert isinstance(n_samples_can, int)

        def canonical_reward_fn(state, action, next_state, done, /):
            """Canonical reward function.
            ``state``, ``action``, ``next_state`` correspond to s, a, s' in the paper
            and ``state_sample``, ``action_sample``, ``next_state_sample``
            correspond to S, A, S' in the paper.

            Args:
                state: The batch of state samples from the coverage distribution.
                action: The batch of action samples from the coverage distribution.
                next_state: The batch of next state samples from the coverage distribution.
                done: The batch of done samples from the coverage distribution.

            Returns:
                The canonicalized reward function.
            """
            n_samples_cov = state.shape[0]

            assert n_samples_cov == action.shape[0] == next_state.shape[0] == done.shape[0]

            n_samples = n_samples_cov * n_samples_can

            # Copy each sample in n_samples_cov to n_samples_can times.
            state_cov, action_cov, next_state_cov, done_cov = utils.broadcast(
                state, action, next_state, done, n_samples_can
            )

            # Create n_samples_cov * n_samples_can samples.
            _, state_sample = self.state_sampler.sample(n_samples)
            action_sample = self.action_sampler.sample(n_samples)
            done_sample, next_state_sample = self.state_sampler.sample(n_samples)

            # Reshape to (n_samples_cov, n_samples_can, -1).
            # This is easier than reshaping the output of each flat reward function call
            # to take the mean along the inner monte carlo estimator.
            state_sample, action_sample, next_state_sample, done_sample = (
                state_sample.reshape(n_samples_cov, n_samples_can, -1),
                action_sample.reshape(n_samples_cov, n_samples_can, -1),
                next_state_sample.reshape(n_samples_cov, n_samples_can, -1),
                done_sample.reshape(n_samples_cov, n_samples_can, -1),
            )

            # E[R(s', A, S')]. We sample action and next state,
            # and pass in ``next_state`` as the state.
            # We make this the first dimension to then take the mean.
            term_1 = np.mean(
                rew_fn(next_state_cov, action_sample, next_state_sample, done_sample, batch_dims=2), axis=0
            )
            # E[R(s, A, S')]. We also sample action and next state.
            # Now it's simply ``state`` that we pass in as the state.
            term_2 = np.mean(rew_fn(state_cov, action_sample, next_state_sample, done_sample, batch_dims=2), axis=0)
            # E[R(S, A, S')]. We sample state, action, and next state.
            # This does not require cartesian product over batches
            # as it's not a random variable.
            term_3 = np.mean(
                rew_fn(
                    state_sample,
                    action_sample,
                    next_state_sample,
                    done_sample,
                    batch_dims=2,
                ),
            )

            return (
                reward_function(
                    state,
                    action,
                    next_state,
                    done,
                ).reshape(n_samples_cov, -1)
                + self.discount_factor * term_1
                - term_2
                - self.discount_factor * term_3
            )

        return canonical_reward_fn


def epic_distance(
    x,
    y,
    /,
    *,
    state_sampler,
    action_sampler,
    discount_factor,
    n_samples_cov: int,
    n_samples_can: int,
):
    """Compute the EPIC distance between two reward functions.

    Helper for automatically instantiating the EPIC Distance class and computing
    the distance between two reward functions.

    Do not use this helper if you want to compute distances between multiple pairs
    of reward functions. Instead, instantiate the EPIC class and call the
    ``distance`` method on the instance multiple times.

    Args:
        x: The first reward function.
        y: The second reward function.
        state_sampler: A sampler for the state space.
        action_sampler: A sampler for the action space.
        discount_factor: The discount factor of the MDP.
        nested: Whether sampling is nested over any expectation operators. See
            ``EPIC.canonicalize`` for more details.

    Returns:
        The EPIC distance between the two reward functions.
    """
    return EPIC(
        state_sampler=state_sampler,
        action_sampler=action_sampler,
        discount_factor=discount_factor,
    ).distance(x, y, n_samples_cov, n_samples_can)
