"""Naive grid-follower: the 'no intelligence' baseline every smarter agent must beat."""
from __future__ import annotations

from ..sim.decision import Decision
from .base import Agent


class NaiveAgent(Agent):
    name = "naive"
    label = "Naive grid-follower"
    description = ("Buys whatever is missing and sells whatever is extra. Never uses the batteries, "
                   "never shifts load, never sends crews. Shows what happens without an orchestrator.")

    def decide_sync(self, obs: dict) -> Decision:
        cur = obs["current"]
        net = cur["net_position_mw"]
        d = Decision(agent=self.name, mode="grid_follow")
        if net >= 0:
            d.sell_mw = min(net, cur["grid"]["export_limit_mw"])
            d.reasons.append(f"Surplus {net:.0f} MW: selling {d.sell_mw:.0f} MW")
        else:
            d.buy_mw = -net
            d.reasons.append(f"Deficit {-net:.0f} MW: buying from the market")
        return d
