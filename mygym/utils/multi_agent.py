from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CTDEDictObservationWrapper(gym.Wrapper):
    """
    Convert the original single-agent flat observation into a CTDE-friendly
    multi-agent dict observation.

    The wrapped environment still receives a joint continuous action vector,
    but the policy gets:
    - ``local_obs``: per-agent local observation for decentralized actors
    - ``global_state``: full building state for the centralized critic
    """

    TIME_VARIABLES = {"month", "day_of_month", "hour"}

    def __init__(
        self,
        env: gym.Env,
        include_agent_id: bool = True,
        temperature_suffix: str = "_air_temperature",
        co2_suffix: str = "_co",
    ) -> None:
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("CTDEDictObservationWrapper expects a flat Box observation space.")
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("CTDEDictObservationWrapper expects a continuous Box action space.")

        observation_variables = list(self.get_wrapper_attr("observation_variables"))
        self.observation_variables = observation_variables
        self.include_agent_id = include_agent_id
        self.temperature_suffix = temperature_suffix
        self.co2_suffix = co2_suffix

        self.temperature_indices = [
            idx for idx, name in enumerate(observation_variables) if name.endswith(self.temperature_suffix)
        ]
        self.co2_indices = [idx for idx, name in enumerate(observation_variables) if name.endswith(self.co2_suffix)]
        self.shared_indices = [
            idx
            for idx, name in enumerate(observation_variables)
            if idx not in self.temperature_indices and idx not in self.co2_indices
        ]

        if not self.temperature_indices:
            raise ValueError("No zone temperature variables were found for MAPPO agent construction.")
        if len(self.temperature_indices) != len(self.co2_indices):
            raise ValueError("Zone temperature and CO2 variable counts must match for per-zone agents.")

        self.n_agents = len(self.temperature_indices)
        self.joint_action_dim = int(np.prod(env.action_space.shape))
        if self.joint_action_dim % self.n_agents != 0:
            raise ValueError("Joint action dimension must be divisible by the number of agents.")
        self.agent_action_dim = self.joint_action_dim // self.n_agents

        self.shared_obs_dim = len(self.shared_indices)
        self.local_obs_dim = self.shared_obs_dim + 2 + (self.n_agents if self.include_agent_id else 0)
        self.agent_ids = np.eye(self.n_agents, dtype=np.float32) if self.include_agent_id else None

        inf = np.finfo(np.float32).max
        self.observation_space = spaces.Dict(
            {
                "local_obs": spaces.Box(
                    low=-inf,
                    high=inf,
                    shape=(self.n_agents, self.local_obs_dim),
                    dtype=np.float32,
                ),
                "global_state": spaces.Box(
                    low=-inf,
                    high=inf,
                    shape=env.observation_space.shape,
                    dtype=np.float32,
                ),
            }
        )

        self.agent_names = [observation_variables[idx] for idx in self.temperature_indices]

    def _build_local_obs(self, obs: np.ndarray) -> np.ndarray:
        shared_obs = obs[self.shared_indices].astype(np.float32, copy=False)
        zone_temps = obs[self.temperature_indices].astype(np.float32, copy=False)
        zone_co2 = obs[self.co2_indices].astype(np.float32, copy=False)

        repeated_shared = np.repeat(shared_obs[None, :], self.n_agents, axis=0)
        local_blocks = [repeated_shared, zone_temps[:, None], zone_co2[:, None]]
        if self.agent_ids is not None:
            local_blocks.append(self.agent_ids)
        return np.concatenate(local_blocks, axis=1).astype(np.float32, copy=False)

    def _convert_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32)
        return {
            "local_obs": self._build_local_obs(obs),
            "global_state": obs.astype(np.float32, copy=False),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_observation(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        transformed_obs = self._convert_observation(obs)
        if "terminal_observation" in info and info["terminal_observation"] is not None:
            info = dict(info)
            info["terminal_observation"] = self._convert_observation(info["terminal_observation"])
        return transformed_obs, reward, terminated, truncated, info
