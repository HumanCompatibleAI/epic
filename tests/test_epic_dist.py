import gym
import numpy as np
import pytest

from epic.distances import epic
from epic.samplers import DummyGymStateSampler, GymSpaceSampler


def rew_fn_1(state, action, next_state, _):
    return state + next_state + action


def rew_fn_2(state, action, next_state, _):
    return state**2 + next_state**2 + 2 * action


def rew_fn_1_potential_shaping(state, action, next_state, _):
    return state + next_state + action + next_state / 3 - state / 3


def test_epic_dist_no_errors():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_2

    dist = epic.EPIC(
        state_sampler=DummyGymStateSampler(space=state_space),
        action_sampler=GymSpaceSampler(space=action_space),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=200, n_samples_can=200)

    assert isinstance(dist, float)


def test_epic_dist_reward_equivalence():
    state_space = gym.spaces.Discrete(10)
    action_space = gym.spaces.Discrete(10)

    x = rew_fn_1
    y = rew_fn_1_potential_shaping

    dist = epic.EPIC(
        state_sampler=DummyGymStateSampler(space=state_space),
        action_sampler=GymSpaceSampler(space=action_space),
        discount_factor=1,
    ).distance(x, y, n_samples_cov=200, n_samples_can=200)

    assert np.isclose(dist, 0, atol=1e-7)
