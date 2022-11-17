"""Implements Divergence-Free Rewards Distance Calculation."""

from typing import Optional, TypeVar, Union, Dict, Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from stable_baselines3.common.logger import make_output_format, Figure

import matplotlib.pyplot as plt

from epic import samplers, types, utils, torch_modules
from epic.distances import base, pearson_mixin

T_co = TypeVar("T_co", covariant=True)


class DivergenceFree(pearson_mixin.PearsonMixin, base.Distance):
    default_samples_cov = 500
    default_samples_can = 500

    def __init__(
        self,
        discount_factor: float,
        coverage_sampler: samplers.BaseSampler[samplers.CoverageSample],
        architecture_hyperparams: Optional[types.PotentialArchitectureHyperparams] = None,
        training_hyperparams: Optional[types.PotentialTrainingHyperparams] = None,
        use_logger: bool = False,
        log_dir: Optional[str] = None,
        log_suffix: str = "",
        store_train_stats: bool = False,
    ):
        """Initialize the Divergence-Free Reward Distance.

        Args:
          coverage_sampler: The sampler for the coverage distribution.
          discount_factor: The discount factor.
          architecture_hyperparams: A dataclass keeping track of different hyperparameters for the neural network architecture.
          training_hyperparams: A dataclass keeping track of different hyperparameters for the neural network training.
          use_logger: Whether or not to configure and use a logger.
          log_dir: The directory to store the log files in.
          log_suffix: A suffix to append to the log files.
          store_train_stats: Whether or not to store statistics from the training run.

        """
        super().__init__(discount_factor, coverage_sampler)

        _, state_sample, _, _ = self.coverage_sampler.sample(1)
        self.state_dim = int(np.prod(state_sample.shape[1:])) if len(state_sample.shape) > 1 else state_sample.shape[0]

        self.architecture_hyperparams = architecture_hyperparams or types.PotentialArchitectureHyperparams(
            hidden_dim=max(4 * self.state_dim, 128)
        )
        self.training_hyperparams = training_hyperparams or types.PotentialTrainingHyperparams()

        self.use_logger = use_logger
        if self.use_logger:
            self.writer = make_output_format("tensorboard", log_dir=log_dir, log_suffix=log_suffix)

        self.store_train_stats = store_train_stats
        if self.store_train_stats:
            self.train_stats = dict()

    def canonicalize(
        self,
        reward_function: types.RewardFunction,
        /,
        n_samples_can: Optional[int],
    ) -> types.RewardFunction:
        """Canonicalizes a reward function into a divergence-free reward
        function of the same equivalence class.

        This is done by fitting a potential function (constructed from a neural
        network) to minimize the L2 norm of the shaped reward function.
        """
        rew_fn = utils.multidim_rew_fn(reward_function)
        n_samples_can = n_samples_can or self.default_samples_can
        assert isinstance(n_samples_can, int)

        net = nn.Sequential(
            *[
                nn.Flatten(),
                nn.Linear(self.state_dim, self.architecture_hyperparams.hidden_dim),
                nn.ReLU(),
                *[
                    torch_modules.Residual(
                        nn.Sequential(
                            nn.LayerNorm(self.architecture_hyperparams.hidden_dim)
                            if self.architecture_hyperparams.use_norm
                            else nn.Identity(),
                            nn.Linear(
                                self.architecture_hyperparams.hidden_dim, self.architecture_hyperparams.hidden_dim
                            ),
                            nn.ReLU(),
                            nn.Linear(
                                self.architecture_hyperparams.hidden_dim, self.architecture_hyperparams.hidden_dim
                            ),
                        ),
                    )
                    for _ in range(self.architecture_hyperparams.depth)
                ],
                nn.ReLU() if self.architecture_hyperparams.depth else nn.Identity(),
                nn.Linear(self.architecture_hyperparams.hidden_dim, 1),
            ],
        )
        device = self.training_hyperparams.device
        net.to(device)

        learning_rate = self.training_hyperparams.learning_rate
        weight_decay = self.training_hyperparams.weight_decay
        max_epochs = self.training_hyperparams.max_epochs

        optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if self.training_hyperparams.use_scheduler:
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda epoch: 0.5 if (epoch > 5000 and epoch < 75000) else 0.25 if (epoch > 7500) else 1.0,
            )
        else:
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda _: 1.0,
            )

        transitions_dataset = torch_modules.TransitionsDataset(
            *(
                self.coverage_sampler.sample(
                    n_samples_can,
                )
            )
        )

        self.training_hyperparams.batch_size = min(self.training_hyperparams.batch_size, n_samples_can)

        batch_size = self.training_hyperparams.batch_size

        losses = []

        def canonical_reward_fn(
            state,
            action,
            next_state,
            done,
            /,
            return_tensor: bool = False,
            device: Union[str, torch.device] = "cpu",
        ):
            """Divergence-Free canonical reward function.

            Args:
                state: The batch of state samples from the coverage distribution.
                action: The batch of action samples from the coverage distribution.
                next_state: The batch of next state samples from the coverage distribution.
                done: The batch of done samples from the coverage distribution.
                return_tensor: Whether to return a torch.Tensor or a numpy nd.array.
                device: The device on which to conduct computations.

            Returns:
                The canonicalized reward function.
            """
            n_samples_cov = state.shape[0]
            assert n_samples_cov == action.shape[0] == next_state.shape[0] == done.shape[0]

            state_tensor = utils.float_tensor_from_numpy(state, device)
            next_state_tensor = utils.float_tensor_from_numpy(next_state, device)
            if state_tensor.ndim == 1:
                assert next_state_tensor.ndim == 1
                state_tensor.unsqueeze_(-1)
                next_state_tensor.unsqueeze_(-1)

            net.to(device)

            shaping = (self.discount_factor * net(next_state_tensor) - net(state_tensor)).squeeze(-1)

            if return_tensor:
                rew_fn_out = utils.float_tensor_from_numpy(rew_fn(state, action, next_state, done), device)
                assert rew_fn_out.ndim == shaping.ndim, "Reward Function's output shouldn't be broadcasted."
                return rew_fn_out + shaping

            rew_fn_out = rew_fn(state, action, next_state, done)
            shaping = utils.numpy_from_tensor(shaping)
            assert rew_fn_out.ndim == shaping.ndim, "Reward Function's output shouldn't be broadcasted."
            return rew_fn_out + shaping

        if self.use_logger:
            self.writer.write(
                dict(
                    training_hyperparams=self.training_hyperparams,
                    architecture_hyperparams=self.architecture_hyperparams,
                ),
                key_excluded=dict(
                    training_hyperparams=(),
                    architecture_hyperparams=(),
                ),
            )

        for _ in tqdm(range(max_epochs)):
            transitions_dataset.shuffle()
            for i in range(len(transitions_dataset) // batch_size):
                action_sample, state_sample, next_state_sample, done_sample = transitions_dataset[
                    i * batch_size : (i + 1) * batch_size
                ]

                l2_loss = torch.mean(
                    (
                        canonical_reward_fn(
                            state_sample,
                            action_sample,
                            next_state_sample,
                            done_sample,
                            return_tensor=True,
                            device=device,
                        )
                    )
                    ** 2,
                )
                l2_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(l2_loss.item())

            # Early stopping if loss has stopped fluctuating
            if self.training_hyperparams.early_stopping:
                if len(losses) >= self.training_hyperparams.early_stopping_patience:
                    losses_window = losses[-self.training_hyperparams.early_stopping_patience :]
                    if np.max(losses_window) - np.min(losses_window) < 1e-6:
                        break
            scheduler.step()
        if self.use_logger:
            self.writer.write(dict(losses=losses), key_excluded=dict(losses=()))
            fig = plt.figure()
            fig.add_subplot().plot(losses, label="Loss", color="blue")
            fig.add_subplot().set_xlabel("Epochs")
            fig.add_subplot().set_ylabel("Loss")
            figure = Figure(figure=fig, close=True)
            self.writer.write(dict(figure=figure), key_excluded=dict(figure=()))

        if self.store_train_stats:
            self.train_stats[f"losses_{len(self.train_stats.keys())}"] = losses

        return canonical_reward_fn


def divergence_free_distance(
    x,
    y,
    /,
    *,
    coverage_sampler,
    discount_factor,
    n_samples_cov: int,
    n_samples_can: int,
):
    """Calculates the divergence-free reward distance between two reward functions.

    Helper function that automatically instantiates the DivergenceFree class and computes the distance
    between two reward functions using its canonicalization.

    Args:
      x: The first reward function.
      y: The second reward function.
      coverage_sampler: The sampler for the coverage distribution.
      discount_factor: The discount factor.
      n_samples_cov: The number of samples to use for the coverage distance.
      n_samples_can: The number of samples to use for the canonicalization.
    """
    return DivergenceFree(discount_factor, coverage_sampler).distance(
        x,
        y,
        n_samples_cov,
        n_samples_can,
    )
