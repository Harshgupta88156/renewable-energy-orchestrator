"""Request bodies for the REST API (validated by FastAPI, documented at /docs)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    scenario: str | None = Field(None, description="Scenario id (see GET /api/meta). Default: storm_alert")
    seed: int | None = Field(None, description="Random seed; same seed = identical day")
    agent: str | None = Field(None, description="Agent name: naive | rule_based | ...")
    days: int = Field(1, ge=1, le=7)
    speed: float | None = Field(None, ge=1, le=20_000, description="Sim seconds per real second (300 = 3 s/step)")
    autostart: bool = False


class SpeedRequest(BaseModel):
    speed: float = Field(..., ge=1, le=20_000)


class AgentRequest(BaseModel):
    agent: str


class InstructionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=300, description="Plain-language guidance for the agent")


class StepRequest(BaseModel):
    steps: int = Field(1, ge=1, le=672)


class EventRequest(BaseModel):
    type: str = Field(..., description="Event type, e.g. cloud_cover, storm, asset_fault (see GET /api/meta)")
    params: dict[str, Any] = Field(default_factory=dict, description="Overrides for the catalog defaults")
    start_in_steps: int = Field(0, ge=0, le=672)
    duration_steps: int | None = Field(-1, description="-1 = catalog default, null = open-ended")
    announce_in_steps: int | None = Field(None, ge=0, description="When the agent learns about it (null = catalog default)")
    severity: str | None = None


class LabRequest(BaseModel):
    scenario: str = "chaos_monkey"
    agents: list[str] = Field(default_factory=lambda: ["naive", "rule_based"], min_length=1)
    runs: int = Field(20, ge=1, le=500)
    days: int = Field(1, ge=1, le=7)
    base_seed: int = 1000
    randomize: bool = True
