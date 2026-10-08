"""Agent registry. Add new agents (optimizer, LLM planner) here and they appear in the API,
the dashboard agent picker, the CLI and the scenario lab automatically."""
from __future__ import annotations

from .base import Agent
from .llm_planner import LLMPlannerAgent
from .naive import NaiveAgent
from .rule_based import RuleBasedAgent

AGENT_CLASSES: dict[str, type[Agent]] = {cls.name: cls for cls in (NaiveAgent, RuleBasedAgent, LLMPlannerAgent)}
FALLBACK_AGENT = "rule_based"


def create_agent(name: str) -> Agent:
    if name not in AGENT_CLASSES:
        raise KeyError(f"Unknown agent '{name}'. Known: {sorted(AGENT_CLASSES)}")
    return AGENT_CLASSES[name]()


def list_agents() -> list[dict]:
    return [cls().info() for cls in AGENT_CLASSES.values()]


__all__ = ["Agent", "AGENT_CLASSES", "FALLBACK_AGENT", "create_agent", "list_agents"]
