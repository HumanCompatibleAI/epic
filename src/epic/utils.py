from typing import Dict, Protocol, TypedDict, Optional, Union

import numpy as np
import torch
from typing_extensions import ParamSpec

from epic import types


def keywordize_rew_fn(reward_fn: types.RewardFunction):
    """Converts a reward function that takes positional arguments to one that takes keyword arguments"""

    def wrapper(
        *,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ):
        return reward_fn(state, action, next_state, done)

    return wrapper


P = ParamSpec("P")


class Fn(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        ...


def multidim_batch_call(function: Fn, arguments: Dict[str, np.ndarray], num_batch_dims: int):
    """Allows converting a function that only processes calls with a single batch dimension
    into a function that processes calls with multiple batch dimensions.

    This is done by flattening the arguments into a single batch dimension, and then
    unflattening (reshaping) the results.
    """
    assert num_batch_dims > 0
    flattened_arguments = {}
    for key, value in arguments.items():
        flattened_arguments[key] = value.reshape(-1, *value.shape[num_batch_dims:])
    result = function(**flattened_arguments)
    return result.reshape(*next(iter(arguments.values())).shape[:num_batch_dims], *result.shape[1:])


def multidim_rew_fn(function: types.RewardFunction):
    """Allows calling a reward function with multidimensional batches.

     This is done by adding an optional keyword argument `batch_dims` specifying the
     number of dimensions of the batch to be processed. If `batch_dims` is not specified,
    the function is called with a single batch dimension.

    The function is then called with the flattened arguments, and the result is reshaped
    to have the original batch dimensions.

    This is useful for reward functions that only process calls with a single batch dimension,
    but are called with multiple batch dimensions (e.g. for taking averages).

    Args:
        function: The reward function to be called.

    Returns:
        A reward function that can be called with multiple batch dimensions.
    """
    function = keywordize_rew_fn(function)

    def wrapper(state, action, next_state, done, batch_dims: int = 1):
        return multidim_batch_call(
            function,
            dict(state=state, action=action, next_state=next_state, done=done),
            num_batch_dims=batch_dims,
        )

    return wrapper


def broadcast(state, action, next_state, done, /, n_samples_can):
    """Tiles the state, action, next_state, and done arrays along the first axis"""
    return (
        np.swapaxes(np.broadcast_to(state, (n_samples_can, *state.shape)), 1, 0),
        np.swapaxes(np.broadcast_to(action, (n_samples_can, *action.shape)), 1, 0),
        np.swapaxes(np.broadcast_to(next_state, (n_samples_can, *next_state.shape)), 1, 0),
        np.swapaxes(np.broadcast_to(done, (n_samples_can, *done.shape)), 1, 0),
    )


def reshape(state, action, next_state, done, /, n_samples_cov):
    """Reshapes the state, action, next_state, and done arrays to have shape (n_samples_cov, -1)"""
    return (
        state.reshape(n_samples_cov, -1),
        action.reshape(n_samples_cov, -1),
        next_state.reshape(n_samples_cov, -1),
        done.reshape(n_samples_cov, -1),
    )


def float_tensor_from_numpy(arr: np.ndarray, device: Optional[Union[str, torch.device]] = None) -> torch.FloatTensor:
    """Converts an np.ndarray to a FloatTensor and moves it to the appropriate
    device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.from_numpy(arr).float().to(device)


def numpy_from_tensor(tensor: torch.Tensor):
    """Converts a Tensor to an np.ndarray."""
    return tensor.cpu().detach().numpy()
