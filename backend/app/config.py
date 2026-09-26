"""Runtime settings (override with environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    db_path: str = _env("REO_DB_PATH", str(BACKEND_DIR / "data" / "reo.sqlite3"))
    persist: bool = _env("REO_PERSIST", "1") == "1"            # write runs/ticks to SQLite
    default_scenario: str = _env("REO_SCENARIO", "storm_alert")
    default_agent: str = _env("REO_AGENT", "rule_based")
    default_speed: float = float(_env("REO_SPEED", "300"))     # sim seconds per real second (300x = 3 s per step)
    decision_timeout_s: float = float(_env("REO_DECISION_TIMEOUT", "8"))
    cors_origins: list[str] = field(default_factory=lambda: _env("REO_CORS", "*").split(","))
    max_lab_runs: int = int(_env("REO_MAX_LAB_RUNS", "500"))


settings = Settings()
