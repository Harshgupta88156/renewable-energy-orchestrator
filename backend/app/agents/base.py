"""Agent interface. Every decision maker (rules, optimizer, LLM planner) implements this.

    obs (dict)  --decide-->  Decision

`obs` is exactly what `Simulator.observe()` returns (also served at GET /api/observation),
so an agent can be developed and unit-tested against saved observations.
Implement `decide_sync` for fast CPU-bound agents, or override `decide` for async ones
(e.g. an LLM call). The runtime wraps every call with a timeout and falls back to the
rule-based agent if it fails, so a broken agent can never take the grid down.
"""
from __future__ import annotations

import asyncio
from abc import ABC

from ..sim.decision import Decision


class Agent(ABC):
    name: str = "base"
    label: str = "Base agent"
    description: str = ""

    def reset(self) -> None:
        """Called when a new run starts."""

    def decide_sync(self, obs: dict) -> Decision:
        raise NotImplementedError

    async def decide(self, obs: dict) -> Decision:
        # run CPU-bound agents off the event loop so the WebSocket stays responsive
        return await asyncio.to_thread(self.decide_sync, obs)

    @property
    def is_sync(self) -> bool:
        """True if the agent implements decide_sync (fast path for batch simulation)."""
        return type(self).decide_sync is not Agent.decide_sync

    def info(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description}
