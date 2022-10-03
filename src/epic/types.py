import abc
from typing import Protocol

import numpy as np
import numpy.typing as npt


class RewardFunction(Protocol):
    """Abstract class for reward function.
    Requires implementation of __call__() to compute the reward given a batch of
    states, actions, and next states.
    """

    def __call__(
        self,
        state: npt.NDArray,
        action: npt.NDArray,
        next_state: npt.NDArray,
        done: npt.NDArray[np.bool_],
        /,
    ) -> npt.NDArray:
        """Compute rewards for a batch of transitions.
        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
        Returns:
            Computed rewards of shape `(batch_size,`).
        """
