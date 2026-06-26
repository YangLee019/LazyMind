from __future__ import annotations

import time
from typing import Any


class ALFWorldTool:
    """Tool wrapper exposing ALFWorld as single-action interactions."""

    __public_apis__ = ['alfworld_step']

    def __init__(self, env: Any):
        self.env = env
        self.max_steps: int | None = None
        self.current_observation: str | None = None
        self.reward: float = 0.0
        self.done: bool = False
        self.info: dict[str, Any] = {}
        self.gamefile: str | None = None
        self.step_count: int = 0
        self.tool_events: list[dict[str, Any]] = []

    def reset(self) -> dict[str, Any]:
        """Reset the current ALFWorld task and return the initial observation."""
        obs, info = self.env.reset()
        self.current_observation = self._first(obs)
        self.reward = 0.0
        self.done = False
        self.info = self._first(info, default={})
        self.gamefile = self._extract_gamefile(self.info)
        self.step_count = 0
        self.tool_events = []
        return {
            'observation': self.current_observation,
            'done': False,
        }

    def step(self, action: str) -> dict[str, Any]:
        """Run one ALFWorld action.

        ALFWorld expects a list of actions, even for batch_size=1.
        """
        if self.max_steps is not None and self.step_count >= self.max_steps:
            raise RuntimeError(f'ALFWorld max_steps exceeded: {self.max_steps}')

        normalized_action = str(action or '').strip()
        if not normalized_action:
            raise ValueError('action must be a non-empty string')

        obs, rewards, dones, infos = self.env.step([normalized_action])
        self.current_observation = self._first(obs)
        self.reward = float(self._first(rewards, default=0.0) or 0.0)
        self.done = bool(self._first(dones, default=False))
        self.info = self._first(infos, default={})
        next_gamefile = self._extract_gamefile(self.info)
        if next_gamefile:
            self.gamefile = next_gamefile
        elif self.gamefile:
            self.info = dict(self.info)
            self.info['extra.gamefile'] = [self.gamefile]
        self.step_count += 1
        event = {
            'step': self.step_count,
            'action': normalized_action,
            'observation': self.current_observation,
            'reward': self.reward,
            'done': self.done,
            'info': self.info,
            'time': time.time(),
        }
        self.tool_events.append(event)
        res = {
            'observation': ''.join(self.current_observation),
            # 'reward': self.reward,
            'done': self.done,
            'admissible_actions': self.info['admissible_commands'],
        }
        return res

    def status(self) -> dict[str, Any]:
        """Return the latest ALFWorld task state."""
        return {
            'observation': self.current_observation,
            'done': self.done,
            'step_count': self.step_count,
        }

    def alfworld_step(self, action: str) -> dict[str, Any]:
        """Execute exactly one ALFWorld action, such as `go to countertop 1`."""
        return self.step(action)

    def alfworld_status(self) -> dict[str, Any]:
        """Return current ALFWorld observation, done flag, and step count."""
        return self.status()

    @staticmethod
    def _first(value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    @staticmethod
    def _extract_gamefile(info: Any) -> str | None:
        if not isinstance(info, dict):
            return None
        value = info.get('extra.gamefile') or info.get('gamefile')
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        text = str(value or '').strip()
        return text or None
