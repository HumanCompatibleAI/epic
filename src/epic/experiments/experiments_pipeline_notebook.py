# %%
import random
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor, RandomPolicy
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy

from epic.distances import divergence_free, epic
from epic import samplers, utils, types
from epic.types import RewardFunction
from epic.vendors import imitation as epic_imitation

import numpy as np

# %%
# Train reward models on various environments and reward functions using imitation

rng = np.random.default_rng(0)

venv = make_vec_env("Pendulum-v1", rng=rng)

reward_net_0 = BasicRewardNet(venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm)
reward_net_1 = BasicRewardNet(venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm)

reward_net_0.train()
reward_net_1.train()

fragmenter = preference_comparisons.RandomFragmenter(
    warning_threshold=0,
    rng=rng,
)
gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
preference_model_0 = preference_comparisons.PreferenceModel(reward_net_0)
preference_model_1 = preference_comparisons.PreferenceModel(reward_net_1)
reward_trainer_0 = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model_0,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=3,
    rng=rng,
)
reward_trainer_1 = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model_0,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=3,
    rng=rng,
)

agent_0 = PPO(
    policy=FeedForward32Policy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=venv,
    seed=0,
    n_steps=2048 // venv.num_envs,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)
agent_1 = PPO(
    policy=ActorCriticPolicy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=venv,
    seed=0,
    n_steps=2048 // venv.num_envs,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)

trajectory_generator_0 = preference_comparisons.AgentTrainer(
    algorithm=agent_0,
    reward_fn=reward_net_0,
    venv=venv,
    exploration_frac=0.0,
    rng=rng,
)
trajectory_generator_1 = preference_comparisons.AgentTrainer(
    algorithm=agent_1,
    reward_fn=reward_net_1,
    venv=venv,
    exploration_frac=0.0,
    rng=rng,
)

pref_comparisons_0 = preference_comparisons.PreferenceComparisons(
    trajectory_generator_0,
    reward_net_0,
    num_iterations=5,
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer_0,
    fragment_length=100,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=1,
)
pref_comparisons_1 = preference_comparisons.PreferenceComparisons(
    trajectory_generator_1,
    reward_net_1,
    num_iterations=5,
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer_1,
    fragment_length=100,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=1,
)
# %%
pref_comparisons_0.train(
    total_timesteps=1_000_000,  # For good performance this should be 1_000_000
    total_comparisons=5_000,  # For good performance this should be 5_000
)
pref_comparisons_1.train(
    total_timesteps=1_000_000,  # For good performance this should be 1_000_000
    total_comparisons=5_000,  # For good performance this should be 5_000
)

# learned_reward_venv_0 = RewardVecEnvWrapper(venv, reward_net_0.predict)
# learned_reward_venv_1 = RewardVecEnvWrapper(venv, reward_net_0.predict)
# %%
# learner_0 = PPO(
#     policy=MlpPolicy,
#     env=learned_reward_venv_0,
#     seed=0,
#     batch_size=64,
#     ent_coef=0.0,
#     learning_rate=0.0003,
#     n_epochs=10,
#     n_steps=64,
# )
# learner_0.learn(100000)
# learner_1 = PPO(
#     policy=MlpPolicy,
#     env=learned_reward_venv_1,
#     seed=0,
#     batch_size=64,
#     ent_coef=0.0,
#     learning_rate=0.0003,
#     n_epochs=10,
#     n_steps=64,
# )
# learner_1.learn(100000)
# %%
# reward_0, _ = evaluate_policy(learner_0, learned_reward_venv_0, n_eval_episodes=10)
# reward_1, _ = evaluate_policy(learner_1, learned_reward_venv_1, n_eval_episodes=10)
# print(f"Reward 0: {reward_0}")
# print(f"Reward 1: {reward_1}")

reward_net_0.eval()
reward_net_1.eval()
# %%
# Collect trajectories from different coverage distributions (expert, random, different mixtures)
# %%
# Measure pairwise div-free and EPIC distances between reward models
reward_0_fn = epic_imitation.reward_net_to_fn(reward_net_0)
reward_1_fn = epic_imitation.reward_net_to_fn(reward_net_1)

dist_div_free = divergence_free.DivergenceFree(
    coverage_sampler=samplers.ProductDistrCoverageSampler(
        samplers.GymSpaceSampler(venv.action_space),
        samplers.DummyGymStateSampler(venv.observation_space),
    ),
    discount_factor=1.0,
    training_hyperparams=types.PotentialTrainingHyperparams(),
).distance(reward_0_fn, reward_1_fn, n_samples_cov=60000, n_samples_can=100000)
print(f"Div-free distance: {dist_div_free}")
# %%
reward_0_fn_epic = epic_imitation.reward_net_to_fn(reward_net_0, device="cpu")
reward_1_fn_epic = epic_imitation.reward_net_to_fn(reward_net_1, device="cpu")
dist_epic = epic.EPIC(
    action_sampler=samplers.GymSpaceSampler(venv.action_space),
    state_sampler=samplers.DummyGymStateSampler(venv.observation_space),
    discount_factor=1.0,
).distance(reward_0_fn_epic, reward_1_fn_epic, n_samples_cov=400, n_samples_can=400)
print(f"EPIC distance: {dist_epic}")
# %%

# Train policies on all reward models and measure how well they perform in terms of each other reward model (empirically determine regret)

# Plot regret vs distance in a scatterplot (with one point for each reward model pair)
