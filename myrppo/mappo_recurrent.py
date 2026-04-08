from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from myrppo.common.buffers import RolloutBuffer
from myrppo.common.callbacks import BaseCallback
from myrppo.common.on_policy_algorithm import OnPolicyAlgorithm
from myrppo.common.policies import BasePolicy
from myrppo.common.type_aliases import GymEnv, MaybeCallback, Schedule
from myrppo.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from myrppo.common.vec_env import VecEnv
from myrppo.mappo_buffers import MAPPODictRolloutBuffer
from myrppo.mappo_policies import MAPPOActorCriticPolicy
from myrppo.type_aliases import RNNStates

SelfRecurrentMAPPO = TypeVar("SelfRecurrentMAPPO", bound="RecurrentMAPPO")


class RecurrentMAPPO(OnPolicyAlgorithm):
    """
    MAPPO with LSTM actors and a centralized LSTM critic.

    Environment interaction still uses a joint action vector, but observations
    are expected to be wrapped as:
    - ``local_obs``: (n_agents, local_obs_dim)
    - ``global_state``: (global_state_dim,)
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpMAPPOLstmPolicy": MAPPOActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[MAPPOActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states: Optional[RNNStates] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("RecurrentMAPPO requires a Dict observation space from the CTDE wrapper.")

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, MAPPOActorCriticPolicy):
            raise ValueError("Policy must subclass MAPPOActorCriticPolicy.")

        actor_hidden_shape = (
            self.policy.n_lstm_layers,
            self.n_envs,
            self.policy.n_agents,
            self.policy.lstm_hidden_size,
        )
        critic_hidden_shape = (
            self.policy.n_lstm_layers,
            self.n_envs,
            self.policy.lstm_hidden_size,
        )

        self._last_lstm_states = RNNStates(
            (
                th.zeros(actor_hidden_shape, device=self.device),
                th.zeros(actor_hidden_shape, device=self.device),
            ),
            (
                th.zeros(critic_hidden_shape, device=self.device),
                th.zeros(critic_hidden_shape, device=self.device),
            ),
        )

        actor_hidden_state_buffer_shape = (
            self.n_steps,
            self.policy.n_lstm_layers,
            self.n_envs,
            self.policy.n_agents,
            self.policy.lstm_hidden_size,
        )
        critic_hidden_state_buffer_shape = (
            self.n_steps,
            self.policy.n_lstm_layers,
            self.n_envs,
            self.policy.lstm_hidden_size,
        )

        self.rollout_buffer = MAPPODictRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            actor_hidden_state_shape=actor_hidden_state_buffer_shape,
            critic_hidden_state_shape=critic_hidden_state_buffer_shape,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass None to disable it."
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(rollout_buffer, MAPPODictRolloutBuffer), "This algorithm requires MAPPODictRolloutBuffer."
        assert self._last_obs is not None, "No previous observation was provided."
        assert self._last_lstm_states is not None, "LSTM states are not initialized."

        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, next_lstm_states = self.policy.forward(
                    obs_tensor, lstm_states, episode_starts
                )

            actions_np = actions.cpu().numpy()
            clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_critic_states = (
                            next_lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            next_lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts_tensor = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_critic_states, episode_starts_tensor
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = next_lstm_states
            lstm_states = next_lstm_states

        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                mask = rollout_data.mask > 1e-8
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfRecurrentMAPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentMAPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentMAPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_last_lstm_states"]
