import numpy as np
import torch
import pytest

from epic import utils


def test_multidim_rew_fn():
    def rew_fn(state, action, next_state, done):
        return state + action + next_state + done

    multidim_rew_fn = utils.multidim_rew_fn(rew_fn)

    assert multidim_rew_fn(np.array([1]), np.array([2]), np.array([3]), np.array([4])) == 10
    assert (
        multidim_rew_fn(np.array([1, 2]), np.array([2, 3]), np.array([3, 4]), np.array([4, 5])) == np.array([10, 14])
    ).all()


def test_keywordize_rew_fn():
    def rew_fn(state, action, next_state, done):
        return state + action + next_state + done

    keywordized_rew_fn = utils.keywordize_rew_fn(rew_fn)

    assert (
        keywordized_rew_fn(
            state=np.array([1]),
            action=np.array([2]),
            next_state=np.array([3]),
            done=np.array([4]),
        )
        == 10
    )


def get_random_arr(dims: int):
    if dims == 1:
        return np.random.rand(np.random.randint(1, 10))
    elif dims == 2:
        return np.random.rand(np.random.randint(1, 10), np.random.randint(1, 10))
    elif dims == 3:
        return np.random.rand(np.random.randint(1, 10), np.random.randint(1, 10), np.random.randint(1, 10))
    else:
        raise ValueError("dims must be 1, 2, or 3")


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("n_samples_can", [2, 3])
def test_broadcast_rew_fn(n_dims, n_samples_can):
    def rew_fn(state, action, next_state, done):
        return state + action + next_state + done

    state = get_random_arr(n_dims)
    action = get_random_arr(n_dims)
    next_state = get_random_arr(n_dims)
    done = get_random_arr(n_dims)

    state_b, action_b, next_state_b, done_b = utils.broadcast(
        state, action, next_state, done, n_samples_can=n_samples_can
    )

    def transformed_shape(arr):
        return arr.shape[0], n_samples_can, *arr.shape[1:]

    assert state_b.shape == transformed_shape(state)
    assert action_b.shape == transformed_shape(action)
    assert next_state_b.shape == transformed_shape(next_state)
    assert done_b.shape == transformed_shape(done)

    # check that entries are equal along the second axis
    assert (state_b[:, 0] == state_b[:, 1]).all()
    assert (action_b[:, 0] == action_b[:, 1]).all()
    assert (next_state_b[:, 0] == next_state_b[:, 1]).all()
    assert (done_b[:, 0] == done_b[:, 1]).all()


def test_tensor_from_numpy():
    arr = np.random.randn(2, 7)

    cpu_tens = utils.float_tensor_from_numpy(arr, "cpu")

    assert cpu_tens.shape == arr.shape
    assert cpu_tens.device == torch.device("cpu")

    if torch.cuda.is_available():
        gpu_tens = utils.float_tensor_from_numpy(arr, "cuda")

        assert gpu_tens.shape == arr.shape
        assert gpu_tens.device.type == "cuda"
        assert (gpu_tens.cpu() == cpu_tens).all()
        assert (cpu_tens.cuda() == gpu_tens).all()


def test_numpy_from_tensor():
    cpu_tensor = torch.randn(2, 7)

    cpu_arr = utils.numpy_from_tensor(cpu_tensor)

    assert cpu_arr.shape == cpu_tensor.shape

    if torch.cuda.is_available():
        gpu_tensor = cpu_tensor.cuda()

        gpu_arr = utils.numpy_from_tensor(gpu_tensor)

        assert np.array_equal(cpu_arr, gpu_arr)
