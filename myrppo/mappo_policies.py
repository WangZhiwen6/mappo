from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from myrppo.common.distributions import DiagGaussianDistribution, Distribution, make_proba_distribution
from myrppo.common.policies import BasePolicy
from myrppo.common.torch_layers import FlattenExtractor, create_mlp
from myrppo.common.type_aliases import PyTorchObs, Schedule
from myrppo.common.utils import zip_strict
from myrppo.type_aliases import RNNStates


class MAPPOActorCriticPolicy(BasePolicy):
    """
    Shared-parameter actor with per-agent LSTM memory and a centralized critic
    over the full building state.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        log_std_init: float = 0.0,
        features_extractor_class: type[nn.Module] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        lstm_kwargs: Optional[dict[str, Any]] = None,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("MAPPOActorCriticPolicy requires a Dict observation space.")
        if not isinstance(action_space, spaces.Box):
            raise TypeError("MAPPOActorCriticPolicy currently supports only continuous Box actions.")
        if "local_obs" not in observation_space.spaces or "global_state" not in observation_space.spaces:
            raise KeyError("Observation space must contain 'local_obs' and 'global_state'.")

        optimizer_kwargs = optimizer_kwargs or {}
        if optimizer_class == th.optim.Adam and "eps" not in optimizer_kwargs:
            optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        local_space = observation_space.spaces["local_obs"]
        global_space = observation_space.spaces["global_state"]
        if len(local_space.shape) != 2 or len(global_space.shape) != 1:
            raise ValueError("Expected local_obs=(n_agents, local_dim) and global_state=(global_dim,).")

        self.n_agents = local_space.shape[0]
        self.local_obs_dim = local_space.shape[1]
        self.global_state_dim = global_space.shape[0]
        self.action_dim = int(np.prod(action_space.shape))
        if self.action_dim % self.n_agents != 0:
            raise ValueError("Joint action dimension must be divisible by the number of agents.")
        self.agent_action_dim = self.action_dim // self.n_agents

        if net_arch is None:
            actor_net_arch = [128, 128]
            critic_net_arch = [128, 128]
        elif isinstance(net_arch, dict):
            actor_net_arch = net_arch.get("pi", [128, 128])
            critic_net_arch = net_arch.get("vf", [128, 128])
        else:
            actor_net_arch = net_arch
            critic_net_arch = net_arch

        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.use_sde = use_sde
        self.full_std = full_std
        self.use_expln = use_expln
        self.share_features_extractor = share_features_extractor
        self.log_std_init = log_std_init
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.lstm_kwargs = lstm_kwargs or {}
        self.actor_net_arch = actor_net_arch
        self.critic_net_arch = critic_net_arch

        self.lstm_actor = nn.LSTM(
            self.local_obs_dim,
            self.lstm_hidden_size,
            num_layers=self.n_lstm_layers,
            **self.lstm_kwargs,
        )
        self.lstm_critic = nn.LSTM(
            self.global_state_dim,
            self.lstm_hidden_size,
            num_layers=self.n_lstm_layers,
            **self.lstm_kwargs,
        )

        actor_latent_dim = actor_net_arch[-1] if len(actor_net_arch) > 0 else self.lstm_hidden_size
        critic_latent_dim = critic_net_arch[-1] if len(critic_net_arch) > 0 else self.lstm_hidden_size

        self.actor_mlp = nn.Sequential(
            *create_mlp(self.lstm_hidden_size, actor_latent_dim, actor_net_arch, activation_fn)
        )
        self.critic_mlp = nn.Sequential(
            *create_mlp(self.lstm_hidden_size, critic_latent_dim, critic_net_arch, activation_fn)
        )

        self.action_net = nn.Linear(actor_latent_dim, self.agent_action_dim)
        self.value_net = nn.Linear(critic_latent_dim, 1)

        self.action_dist = make_proba_distribution(action_space, use_sde=False, dist_kwargs=None)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise NotImplementedError("MAPPOActorCriticPolicy currently supports only diagonal Gaussian actions.")
        self.log_std = nn.Parameter(th.ones(self.action_dim) * self.log_std_init, requires_grad=True)

        if self.ortho_init:
            module_gains = {
                self.lstm_actor: 1.0,
                self.lstm_critic: 1.0,
                self.actor_mlp: 1.0,
                self.critic_mlp: 1.0,
                self.action_net: 0.01,
                self.value_net: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(lambda m, g=gain: self.init_weights(m, g))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                lr_schedule=self._dummy_schedule,
                net_arch={"pi": self.actor_net_arch, "vf": self.critic_net_arch},
                activation_fn=self.activation_fn,
                ortho_init=self.ortho_init,
                log_std_init=self.log_std_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                lstm_hidden_size=self.lstm_hidden_size,
                n_lstm_layers=self.n_lstm_layers,
                lstm_kwargs=self.lstm_kwargs,
            )
        )
        return data

    @property
    def actor_state_shape(self) -> tuple[int, int, int]:
        return (self.n_lstm_layers, self.n_agents, self.lstm_hidden_size)

    @property
    def critic_state_shape(self) -> tuple[int, int]:
        return (self.n_lstm_layers, self.lstm_hidden_size)

    def _process_actor_sequence(
        self,
        local_obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        n_seq = lstm_states[0].shape[1]
        seq_obs = local_obs.reshape((n_seq, -1, self.n_agents, self.local_obs_dim)).swapaxes(0, 1)
        seq_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        hidden_state = lstm_states[0].contiguous()
        cell_state = lstm_states[1].contiguous()

        if th.all(seq_starts == 0.0):
            lstm_output, (hidden_state, cell_state) = self.lstm_actor(
                seq_obs.reshape(seq_obs.shape[0], n_seq * self.n_agents, self.local_obs_dim),
                (
                    hidden_state.reshape(self.n_lstm_layers, n_seq * self.n_agents, self.lstm_hidden_size),
                    cell_state.reshape(self.n_lstm_layers, n_seq * self.n_agents, self.lstm_hidden_size),
                ),
            )
            lstm_output = lstm_output.reshape(seq_obs.shape[0], n_seq, self.n_agents, self.lstm_hidden_size)
            lstm_output = lstm_output.transpose(0, 1).reshape(-1, self.n_agents, self.lstm_hidden_size)
            hidden_state = hidden_state.reshape(self.n_lstm_layers, n_seq, self.n_agents, self.lstm_hidden_size)
            cell_state = cell_state.reshape(self.n_lstm_layers, n_seq, self.n_agents, self.lstm_hidden_size)
            return lstm_output, (hidden_state, cell_state)

        outputs = []
        for local_step, episode_start in zip_strict(seq_obs, seq_starts):
            reset_mask = (1.0 - episode_start).view(1, n_seq, 1, 1)
            hidden_state = hidden_state * reset_mask
            cell_state = cell_state * reset_mask

            output, (next_hidden, next_cell) = self.lstm_actor(
                local_step.reshape(1, n_seq * self.n_agents, self.local_obs_dim),
                (
                    hidden_state.reshape(self.n_lstm_layers, n_seq * self.n_agents, self.lstm_hidden_size),
                    cell_state.reshape(self.n_lstm_layers, n_seq * self.n_agents, self.lstm_hidden_size),
                ),
            )
            outputs.append(output.reshape(1, n_seq, self.n_agents, self.lstm_hidden_size))
            hidden_state = next_hidden.reshape(self.n_lstm_layers, n_seq, self.n_agents, self.lstm_hidden_size)
            cell_state = next_cell.reshape(self.n_lstm_layers, n_seq, self.n_agents, self.lstm_hidden_size)

        actor_output = th.cat(outputs, dim=0).transpose(0, 1).reshape(-1, self.n_agents, self.lstm_hidden_size)
        return actor_output, (hidden_state, cell_state)

    def _process_critic_sequence(
        self,
        global_state: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        n_seq = lstm_states[0].shape[1]
        features_sequence = global_state.reshape((n_seq, -1, self.global_state_dim)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = self.lstm_critic(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        outputs = []
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = self.lstm_critic(
                features.unsqueeze(dim=0),
                (
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            outputs.append(hidden)

        critic_output = th.flatten(th.cat(outputs).transpose(0, 1), start_dim=0, end_dim=1)
        return critic_output, lstm_states

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        batch_size = latent_pi.shape[0]
        per_agent_latent = latent_pi.reshape(batch_size * self.n_agents, -1)
        mean_actions = self.action_net(per_agent_latent).reshape(batch_size, self.action_dim)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _actor_latent(self, local_obs: th.Tensor, lstm_states: tuple[th.Tensor, th.Tensor], episode_starts: th.Tensor):
        actor_output, next_actor_states = self._process_actor_sequence(local_obs, lstm_states, episode_starts)
        actor_latent = self.actor_mlp(actor_output.reshape(-1, self.lstm_hidden_size))
        actor_latent = actor_latent.reshape(-1, self.n_agents, actor_latent.shape[-1])
        return actor_latent, next_actor_states

    def _critic_latent(
        self,
        global_state: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ):
        critic_output, next_critic_states = self._process_critic_sequence(global_state, lstm_states, episode_starts)
        critic_latent = self.critic_mlp(critic_output)
        return critic_latent, next_critic_states

    def forward(
        self,
        obs: PyTorchObs,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        assert isinstance(obs, dict), "MAPPOActorCriticPolicy expects dict observations."

        local_obs = obs["local_obs"].float()
        global_state = obs["global_state"].float()

        actor_latent, next_actor_states = self._actor_latent(local_obs, lstm_states.pi, episode_starts)
        critic_latent, next_critic_states = self._critic_latent(global_state, lstm_states.vf, episode_starts)

        distribution = self._get_action_dist_from_latent(actor_latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(critic_latent)

        return actions, values, log_prob, RNNStates(next_actor_states, next_critic_states)

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        assert isinstance(obs, dict), "MAPPOActorCriticPolicy expects dict observations."

        local_obs = obs["local_obs"].float()
        global_state = obs["global_state"].float()

        actor_latent, _ = self._actor_latent(local_obs, lstm_states.pi, episode_starts)
        critic_latent, _ = self._critic_latent(global_state, lstm_states.vf, episode_starts)

        distribution = self._get_action_dist_from_latent(actor_latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(critic_latent)
        return values, log_prob, entropy

    def predict_values(
        self,
        obs: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        assert isinstance(obs, dict), "MAPPOActorCriticPolicy expects dict observations."
        critic_latent, _ = self._critic_latent(obs["global_state"].float(), lstm_states, episode_starts)
        return self.value_net(critic_latent)

    def get_distribution(
        self,
        obs: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, th.Tensor]]:
        assert isinstance(obs, dict), "MAPPOActorCriticPolicy expects dict observations."
        actor_latent, next_actor_states = self._actor_latent(obs["local_obs"].float(), lstm_states, episode_starts)
        return self._get_action_dist_from_latent(actor_latent), next_actor_states

    def _predict(
        self,
        observation: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        distribution, next_actor_states = self.get_distribution(
            observation, lstm_states=lstm_states, episode_starts=episode_starts
        )
        return distribution.get_actions(deterministic=deterministic), next_actor_states

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation)
        assert isinstance(observation, dict), "MAPPOActorCriticPolicy expects dict observations."

        n_envs = observation["global_state"].shape[0]
        if state is None:
            actor_state = np.zeros((self.n_lstm_layers, n_envs, self.n_agents, self.lstm_hidden_size), dtype=np.float32)
            critic_state = np.zeros((self.n_lstm_layers, n_envs, self.lstm_hidden_size), dtype=np.float32)
            state = (actor_state, actor_state.copy(), critic_state, critic_state.copy())

        if episode_start is None:
            episode_start = np.zeros((n_envs,), dtype=np.float32)

        with th.no_grad():
            states = RNNStates(
                (
                    th.tensor(state[0], dtype=th.float32, device=self.device),
                    th.tensor(state[1], dtype=th.float32, device=self.device),
                ),
                (
                    th.tensor(state[2], dtype=th.float32, device=self.device),
                    th.tensor(state[3], dtype=th.float32, device=self.device),
                ),
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, _, _, next_states = self.forward(
                observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
            )

        next_state = (
            next_states.pi[0].cpu().numpy(),
            next_states.pi[1].cpu().numpy(),
            next_states.vf[0].cpu().numpy(),
            next_states.vf[1].cpu().numpy(),
        )
        actions = actions.cpu().numpy()

        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, next_state
