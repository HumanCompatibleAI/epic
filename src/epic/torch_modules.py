import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import numpy.typing as npt


class Residual(nn.Module):
    """A Wrapper for another PyTorch Module that implements a residual
    connection around that module."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class TransitionsDataset(Dataset):
    """Dataset that stores Transition Data."""

    def __init__(
        self,
        action: npt.NDArray,
        state: npt.NDArray,
        next_state: npt.NDArray,
        done: npt.NDArray,
    ):
        assert state.shape[0] == action.shape[0] == next_state.shape[0] == done.shape[0]
        self.state = state
        self.action = action
        self.next_state = next_state
        self.done = done

    def __getitem__(self, idx):
        return self.action[idx], self.state[idx], self.next_state[idx], self.done[idx]

    def __len__(self):
        return self.state.shape[0]

    def shuffle(self):
        np.random.shuffle(self.action)
        np.random.shuffle(self.state)
        np.random.shuffle(self.next_state)
        np.random.shuffle(self.done)
