"""SQLite audit log: every run, every decision, every event. Enables replay and 'why did it do that?'."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT, scenario TEXT, seed INTEGER, days INTEGER,
    agent TEXT, status TEXT, summary_json TEXT
);
CREATE TABLE IF NOT EXISTS ticks (
    run_id TEXT, step INTEGER, time TEXT, agent TEXT, mode TEXT, fallback INTEGER,
    reasons_json TEXT, tick_json TEXT, PRIMARY KEY (run_id, step)
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT, event_id TEXT, created_step INTEGER, source TEXT, event_json TEXT
);
CREATE TABLE IF NOT EXISTS lab_jobs (
    job_id TEXT PRIMARY KEY, created_at TEXT, status TEXT, request_json TEXT, result_json TEXT
);
"""


def to_json(obj) -> str:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, default=default, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path: str, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _exec(self, sql: str, args: tuple = ()) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    def _query(self, sql: str, args: tuple = ()) -> list[tuple]:
        if not self._conn:
            return []
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # ---- runs
    def create_run(self, run_id: str, scenario: str, seed: int, days: int, agent: str) -> None:
        self._exec("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                   (run_id, _now(), scenario, seed, days, agent, "running", None))

    def finish_run(self, run_id: str, summary: dict, status: str = "finished") -> None:
        self._exec("UPDATE runs SET status=?, summary_json=? WHERE run_id=?", (status, to_json(summary), run_id))

    def add_tick(self, run_id: str, tick: dict) -> None:
        slim = {k: v for k, v in tick.items() if k != "kpi_delta"}
        self._exec("INSERT OR REPLACE INTO ticks VALUES (?,?,?,?,?,?,?,?)",
                   (run_id, tick["step"], tick["time"], tick["agent"], tick["mode"], int(tick["fallback"]),
                    to_json(tick["reasons"]), to_json(slim)))

    def add_event(self, run_id: str, event: dict, created_step: int) -> None:
        self._exec("INSERT INTO events VALUES (?,?,?,?,?)",
                   (run_id, event["id"], created_step, event.get("source", ""), to_json(event)))

    def list_runs(self, limit: int = 50) -> list[dict]:
        rows = self._query("SELECT run_id, created_at, scenario, seed, days, agent, status, summary_json "
                           "FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            summary = json.loads(r[7]) if r[7] else None
            money = (summary or {}).get("kpis", {}).get("money", {})
            energy = (summary or {}).get("kpis", {}).get("energy", {})
            out.append({"run_id": r[0], "created_at": r[1], "scenario": r[2], "seed": r[3], "days": r[4],
                        "agent": r[5], "status": r[6], "profit": money.get("profit"),
                        "unserved_mwh": energy.get("unserved_mwh")})
        return out

    def get_run(self, run_id: str) -> dict | None:
        rows = self._query("SELECT run_id, created_at, scenario, seed, days, agent, status, summary_json "
                           "FROM runs WHERE run_id=?", (run_id,))
        if not rows:
            return None
        r = rows[0]
        return {"run_id": r[0], "created_at": r[1], "scenario": r[2], "seed": r[3], "days": r[4],
                "agent": r[5], "status": r[6], "summary": json.loads(r[7]) if r[7] else None}

    def get_ticks(self, run_id: str, since: int = 0, limit: int = 500) -> list[dict]:
        rows = self._query("SELECT tick_json FROM ticks WHERE run_id=? AND step>=? ORDER BY step LIMIT ?",
                           (run_id, since, limit))
        return [json.loads(r[0]) for r in rows]

    def get_events(self, run_id: str) -> list[dict]:
        return [json.loads(r[0]) for r in self._query(
            "SELECT event_json FROM events WHERE run_id=? ORDER BY created_step", (run_id,))]

    # ---- lab jobs
    def save_lab_job(self, job_id: str, status: str, request: dict, result: dict | None) -> None:
        self._exec("INSERT OR REPLACE INTO lab_jobs VALUES (?,?,?,?,?)",
                   (job_id, _now(), status, to_json(request), to_json(result) if result else None))
