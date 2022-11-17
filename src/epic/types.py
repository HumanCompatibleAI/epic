import abc
from typing import Protocol, Callable, Union
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


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


@dataclass(frozen=True)
class PotentialArchitectureHyperparams:
    """Hyperparameters for the potential function architecture.

    Args:
        depth: Number of Residual MLP layers in the potential function.
        hidden_dim: Dimension of the hidden layers in the potential function.
        use_norm: Whether to include a LayerNorm at the beginning of each Residual Block.
    """

    depth: int = 1
    hidden_dim: int = 128
    use_norm: bool = True


@dataclass
class PotentialTrainingHyperparams:
    """Hyperparameters for training the potential function.

    Args:
        learning_rate: Learning rate for the potential function.
        weight_decay: Weight decay for the potential function.
        max_epochs: Maximum number of epochs to train the potential function.
        batch_size: Batch size for training the potential function.
        use_scheduler: Whether to use a learning rate scheduler.
        device: Device on which to train the potential function.
        early_stopping: Whether to use early stopping.
        early_stopping_patience: Patience for early stopping.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 10000
    batch_size: int = 10000
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    use_scheduler: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 1000
