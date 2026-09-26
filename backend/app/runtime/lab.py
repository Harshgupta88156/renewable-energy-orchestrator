"""Scenario lab (F3): simulate many days with random variations and compare agents fairly.

Every agent sees exactly the same randomised days (same seed -> same weather, prices and
events), so differences come from decisions only. We report distributions (mean, P5, P95),
reliability (days with any load shed) and paired win-rates against the first agent.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable

import numpy as np

from ..agents import FALLBACK_AGENT, create_agent
from ..sim.decision import Decision
from ..sim.engine import Simulator
from ..sim.scenarios import get_scenario

METRICS = {  # name -> (path in kpi summary, higher_is_better)
    "objective_total": (("objective", "total"), False),
    "profit": (("money", "profit"), True),
    "net_energy_cost": (("money", "net_energy_cost"), False),
    "dsm_cost": (("money", "dsm_cost"), False),
    "unserved_mwh": (("energy", "unserved_mwh"), False),
    "curtailed_mwh": (("energy", "curtailed_mwh"), False),
    "co2_t": (("carbon", "co2_t"), False),
    "clean_share_pct": (("percent", "clean_share_pct"), True),
    "renewable_utilization_pct": (("percent", "renewable_utilization_pct"), True),
    "reliability_pct": (("percent", "reliability_pct"), True),
    "emergency_actions": (("agent", "emergency_actions"), False),
    "fallbacks": (("agent", "fallbacks"), False),
}


def _get(d: dict, path: tuple[str, ...]) -> float:
    for p in path:
        d = d[p]
    return float(d)


def run_single(scenario_id: str, agent_name: str, seed: int, days: int = 1, randomize: bool = True) -> dict:
    """Run one full simulation headless and return its KPI summary."""
    sim = Simulator(get_scenario(scenario_id), seed=seed, days=days, randomize=randomize, keep_history=False)
    agent, fallback = create_agent(agent_name), create_agent(FALLBACK_AGENT)
    agent.reset()
    loop = None if agent.is_sync else asyncio.new_event_loop()
    try:
        while not sim.done:
            obs = sim.observe()
            t0 = time.perf_counter()
            fb = False
            try:
                d = agent.decide_sync(obs) if loop is None else loop.run_until_complete(agent.decide(obs))
                if isinstance(d, dict):
                    d = Decision.from_dict(d)
            except Exception as e:  # same safety net as the live runtime
                d = fallback.decide_sync(obs)
                d.reasons.insert(0, f"FALLBACK: {type(e).__name__}")
                fb = True
            try:
                sim.apply_decision(d, latency_ms=(time.perf_counter() - t0) * 1000, fallback=fb)
            except ValueError:  # malformed decision: rejected before any state change
                sim.apply_decision(fallback.decide_sync(obs), fallback=True)
    finally:
        if loop:
            loop.close()
    s = sim.summary()
    s["agent"] = agent_name
    s["n_events"] = len(sim.events)
    return s


def _stats(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def run_batch(scenario_id: str, agents: list[str], runs: int, days: int = 1, base_seed: int = 1000,
              randomize: bool = True, progress: Callable[[int, int], None] | None = None) -> dict:
    get_scenario(scenario_id)
    for a in agents:
        create_agent(a)  # validate names early
    started = time.perf_counter()
    per_agent: dict[str, list[dict]] = {a: [] for a in agents}
    total = runs * len(agents)
    done = 0
    for i in range(runs):
        seed = base_seed + i
        for a in agents:
            per_agent[a].append(run_single(scenario_id, a, seed, days, randomize))
            done += 1
            if progress:
                progress(done, total)

    summary = {}
    for a, results in per_agent.items():
        kpis = [r["kpis"] for r in results]
        summary[a] = {
            "runs": len(results),
            "metrics": {m: _stats([_get(k, path) for k in kpis]) for m, (path, _) in METRICS.items()},
            "days_with_load_shed": int(sum(1 for k in kpis if k["energy"]["unserved_mwh"] > 1e-3)),
            "total_unserved_mwh": float(sum(k["energy"]["unserved_mwh"] for k in kpis)),
        }

    baseline = agents[0]
    comparisons = {}
    for a in agents[1:]:
        comp = {}
        for m, (path, higher) in METRICS.items():
            base_vals = np.array([_get(r["kpis"], path) for r in per_agent[baseline]])
            vals = np.array([_get(r["kpis"], path) for r in per_agent[a]])
            diff = vals - base_vals
            wins = (diff > 1e-9) if higher else (diff < -1e-9)
            ties = np.abs(diff) <= 1e-9
            comp[m] = {"mean_diff": float(diff.mean()), "win_rate": float(wins.mean()),
                       "tie_rate": float(ties.mean())}
        comparisons[a] = {"vs": baseline, **comp}

    return {
        "scenario": scenario_id, "agents": agents, "runs": runs, "days": days,
        "base_seed": base_seed, "randomize": randomize,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "summary": summary, "comparisons": comparisons,
        "per_run": {a: [{"seed": base_seed + i, "profit": r["kpis"]["money"]["profit"],
                         "objective": r["kpis"]["objective"]["total"],
                         "unserved_mwh": r["kpis"]["energy"]["unserved_mwh"],
                         "curtailed_mwh": r["kpis"]["energy"]["curtailed_mwh"],
                         "co2_t": r["kpis"]["carbon"]["co2_t"], "n_events": r["n_events"]}
                        for i, r in enumerate(res)] for a, res in per_agent.items()},
    }


class LabManager:
    """Runs batch jobs in a worker thread so the live simulation keeps ticking."""

    def __init__(self, hub, store, max_runs: int = 500):
        self.hub, self.store, self.max_runs = hub, store, max_runs
        self.jobs: dict[str, dict] = {}

    def submit(self, request: dict) -> dict:
        runs = int(request.get("runs", 20))
        if not 1 <= runs <= self.max_runs:
            raise ValueError(f"runs must be between 1 and {self.max_runs}")
        get_scenario(request["scenario"])
        for a in request["agents"]:
            create_agent(a)
        job_id = uuid.uuid4().hex[:10]
        job = {"job_id": job_id, "status": "running", "request": request, "progress": 0.0, "result": None,
               "error": None}
        self.jobs[job_id] = job
        asyncio.get_running_loop().create_task(self._run(job))
        return {k: v for k, v in job.items() if k != "result"}

    async def _run(self, job: dict) -> None:
        loop = asyncio.get_running_loop()
        req = job["request"]

        def progress(done: int, total: int) -> None:
            job["progress"] = done / total
            loop.call_soon_threadsafe(asyncio.ensure_future, self.hub.broadcast(
                {"type": "lab_progress", "job_id": job["job_id"], "progress": job["progress"]}))

        try:
            result = await asyncio.to_thread(
                run_batch, req["scenario"], req["agents"], int(req.get("runs", 20)), int(req.get("days", 1)),
                int(req.get("base_seed", 1000)), bool(req.get("randomize", True)), progress)
            job.update(status="finished", result=result, progress=1.0)
        except Exception as e:
            job.update(status="failed", error=f"{type(e).__name__}: {e}")
        await asyncio.to_thread(self.store.save_lab_job, job["job_id"], job["status"], req, job["result"])
        await self.hub.broadcast({"type": "lab_done", "job_id": job["job_id"], "status": job["status"]})

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [{k: v for k, v in j.items() if k != "result"} for j in self.jobs.values()]
