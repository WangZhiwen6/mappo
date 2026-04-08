from __future__ import annotations

from collections.abc import Generator
from typing import Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from myrppo.common.buffers import DictRolloutBuffer
from myrppo.common.vec_env import VecNormalize
from myrppo.recurrent.buffers import create_sequencers
from myrppo.type_aliases import RecurrentDictRolloutBufferSamples, RNNStates


class MAPPODictRolloutBuffer(DictRolloutBuffer):
    """
    Recurrent rollout buffer for MAPPO with:
    - actor hidden states per environment and per agent
    - critic hidden states per environment on the centralized global state
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        actor_hidden_state_shape: tuple[int, int, int, int, int],
        critic_hidden_state_shape: tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.actor_hidden_state_shape = actor_hidden_state_shape
        self.critic_hidden_state_shape = critic_hidden_state_shape
        self.seq_start_indices = None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        super().reset()
        self.hidden_states_pi = np.zeros(self.actor_hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.actor_hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.critic_hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.critic_hidden_state_shape, dtype=np.float32)

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())
        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        if not self.generator_ready:
            self.hidden_states_pi = self.hidden_states_pi.swapaxes(1, 2)
            self.cell_states_pi = self.cell_states_pi.swapaxes(1, 2)
            self.hidden_states_vf = self.hidden_states_vf.swapaxes(1, 2)
            self.cell_states_vf = self.cell_states_vf.swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentDictRolloutBufferSamples:
        del env
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        lstm_states_pi = (
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {
            key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()
        }

        return RecurrentDictRolloutBufferSamples(
            observations=observations,
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(
                (
                    self.to_torch(lstm_states_pi[0]).contiguous(),
                    self.to_torch(lstm_states_pi[1]).contiguous(),
                ),
                (
                    self.to_torch(lstm_states_vf[0]).contiguous(),
                    self.to_torch(lstm_states_vf[1]).contiguous(),
                ),
            ),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )
