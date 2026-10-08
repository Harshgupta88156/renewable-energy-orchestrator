"""Live simulation session: the clock, the agent loop, the safety fallback and broadcasting.

    clock tick (every 900/speed real seconds)  or  event injected (wakes the loop immediately)
        -> obs = sim.observe()
        -> decision = agent.decide(obs)   [timeout + exception guard -> rule-based fallback]
        -> tick = sim.apply_decision(decision)
        -> persist to SQLite, broadcast over WebSocket
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..agents import FALLBACK_AGENT, create_agent, list_agents
from ..config import Settings
from ..sim.clock import hhmm, label
from ..sim.constants import STEP_MINUTES
from ..sim.decision import Decision
from ..sim.engine import Simulator, chart_point
from ..sim.events import EVENT_CATALOG
from ..sim.kpis import DEFAULT_WEIGHTS
from ..sim.scenarios import SCENARIOS, get_scenario
from ..store.db import RunStore
from .hub import Hub

log = logging.getLogger("reo.session")

MIN_SPEED, MAX_SPEED = 1.0, 20_000.0


def action_summary(tick: dict) -> list[str]:
    """One-line human-readable actions, for the decision log."""
    a, out = tick["applied"], []
    for bid, b in a["batteries"].items():
        if b["charge_mw"] > 0.5:
            out.append(f"{bid} charge {b['charge_mw']:.0f} MW")
        if b["discharge_mw"] > 0.5:
            out.append(f"{bid} discharge {b['discharge_mw']:.0f} MW")
    m = a["market"]
    if m["buy_mw"] > 0.5:
        out.append(f"Buy {m['buy_mw']:.0f} MW")
    if m["sell_mw"] > 0.5:
        out.append(f"Sell {m['sell_mw']:.0f} MW")
    for k, v in a["curtail_mw"].items():
        out.append(f"Curtail {k} {v:.0f} MW")
    for k, v in a["demand_response_mw"].items():
        out.append(f"DR {k} {v:.0f} MW")
    for k, v in a["shift_mw"].items():
        out.append(f"Shift {k} {v:.0f} MW")
    for k, v in a["noncritical_cut_mw"].items():
        out.append(f"Cut non-critical {k} {v:.0f} MW")
    for mnt in a["maintenance"]:
        if mnt["action"] == "dispatch_inspection":
            out.append(f"Crew -> {mnt['asset_id']}")
        else:
            out.append(f"Maintenance {mnt['maintenance_id']} -> {hhmm(mnt['to_step'])}")
    if tick["flows"]["unserved_mw"] > 0.05:
        out.append(f"SHED {tick['flows']['unserved_mw']:.1f} MW")
    return out or ["Hold"]


def public_tick(tick: dict) -> dict:
    return {k: v for k, v in tick.items() if k != "kpi_delta"}


def catalog() -> dict:
    return {
        "scenarios": [s.to_dict() for s in SCENARIOS.values()],
        "agents": list_agents(),
        "events": {k: {**v, "type": k} for k, v in EVENT_CATALOG.items()},
        "objective_weights": DEFAULT_WEIGHTS,
        "step_minutes": STEP_MINUTES,
        "speed_range": [MIN_SPEED, MAX_SPEED],
    }


class SimulationSession:
    def __init__(self, hub: Hub, store: RunStore, settings: Settings):
        self.hub, self.store, self.settings = hub, store, settings
        self.fallback = create_agent(FALLBACK_AGENT)
        self.speed = settings.default_speed
        self.state = "idle"
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()
        self.sim: Simulator | None = None
        self.agent = None
        self.instructions: list[str] = []
        self.run_id = ""
        self.chart: list[dict] = []
        self.decision_log: list[dict] = []
        # agents and the database get their own threads: a hung agent can't starve DB writes
        self._agent_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent")
        self._db_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db")
        self._inflight = None  # the agent call still running from an earlier (timed-out) step

    async def _db(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._db_pool, fn, *args)

    # ------------------------------------------------------------------ lifecycle
    async def reset(self, scenario: str | None = None, seed: int | None = None, agent: str | None = None,
                    days: int = 1, speed: float | None = None, autostart: bool = False) -> dict:
        sc = get_scenario(scenario or self.settings.default_scenario)  # validate before touching anything
        new_agent = create_agent(agent or self.settings.default_agent)
        new_agent.reset()
        new_sim = Simulator(sc, seed=seed, days=days)
        await self._stop_loop()
        async with self._lock:
            self.agent, self.sim = new_agent, new_sim
            self.instructions = []
            if speed is not None:
                self.speed = float(min(max(speed, MIN_SPEED), MAX_SPEED))
            self.run_id = uuid.uuid4().hex[:12]
            self.chart, self.decision_log = [], []
            self.state = "paused"
            self._inflight = None
        await self._db(self.store.create_run, self.run_id, sc.id, self.sim.seed, self.sim.days, self.agent.name)
        await self.hub.broadcast(self.snapshot())
        if autostart:
            await self.start()
        return self.status()

    async def start(self) -> dict:
        if self.sim is None:
            await self.reset()
        if self.sim.done:
            return self.status()
        self.state = "running"
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="sim-loop")
        await self._broadcast_status()
        return self.status()

    async def pause(self) -> dict:
        if self.state == "running":
            self.state = "paused"
            self._wake.set()
        await self._broadcast_status()
        return self.status()

    async def step(self, n: int = 1) -> dict:
        if self.state == "running":
            await self.pause()
        run_id = self.run_id
        for _ in range(max(1, min(int(n), 96 * 7))):
            if self.sim is None or self.sim.done or self.run_id != run_id or self.state == "running":
                break  # a reset or a start happened meanwhile
            await self._tick(run_id)
        return self.status()

    async def set_speed(self, speed: float) -> dict:
        self.speed = float(min(max(speed, MIN_SPEED), MAX_SPEED))
        self._wake.set()
        await self._broadcast_status()
        return self.status()

    async def set_agent(self, name: str) -> dict:
        self.agent = create_agent(name)
        for note in self.instructions:
            self.agent.instruct(note)
        self.agent.reset()
        await self._broadcast_status()
        return self.status()

    async def instruct(self, text: str) -> dict:
        """Operator note for the agent (e.g. "cyclone warning tonight, protect the hospital feeder")."""
        if self.agent is None:
            await self.reset()
        self.instructions = (self.instructions + [text.strip()[:300]])[-10:]
        res = self.agent.instruct(text)
        await self.hub.broadcast({"type": "instruction", "text": text, "agent": self.agent.name, "result": res})
        return {"agent": self.agent.name, **res}

    async def inject_event(self, etype: str, params: dict | None, start_in_steps: int = 0,
                           duration_steps: int | None | str = "default", announce_in_steps: int | None = None,
                           severity: str | None = None) -> dict:
        if self.sim is None:
            await self.reset()
        async with self._lock:
            if self.sim.done:
                raise ValueError("Simulation finished; reset to start a new run")
            ev = self.sim.inject_event(etype, params, start_in_steps=start_in_steps,
                                       duration_steps=duration_steps, announce_in_steps=announce_in_steps,
                                       severity=severity)
            data = ev.to_dict(self.sim.step)
            run_id, step = self.run_id, self.sim.step
        await self._db(self.store.add_event, run_id, data, step)
        await self.hub.broadcast({"type": "event", "event": data, "observation": self.sim.observe(),
                                  "events": self.sim.event_timeline()})
        if self.state == "running" and ev.start_step <= self.sim.step:
            self._wake.set()  # react now instead of waiting for the next clock tick
        return data

    async def _stop_loop(self) -> None:
        if self._task and not self._task.done():
            self.state = "paused"
            self._wake.set()
            try:
                await asyncio.wait_for(self._task, timeout=30)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._task = None

    async def shutdown(self) -> None:
        await self._stop_loop()
        self._agent_pool.shutdown(wait=False, cancel_futures=True)
        self._db_pool.shutdown(wait=True)

    # ------------------------------------------------------------------ loop
    @property
    def tick_interval_s(self) -> float:
        return STEP_MINUTES * 60 / self.speed

    async def _loop(self) -> None:
        try:
            while self.state == "running" and self.sim and not self.sim.done:
                started = time.perf_counter()
                await self._tick()
                if self.state != "running" or self.sim.done:
                    break
                wait = max(0.0, self.tick_interval_s - (time.perf_counter() - started))
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
        except Exception:  # never let the loop die silently
            log.exception("Simulation loop crashed")
            self.state = "paused"
            await self.hub.broadcast({"type": "error", "message": "Simulation loop crashed; see server log"})
            await self._broadcast_status()

    async def _call_agent(self, obs: dict) -> Decision:
        timeout = self.settings.decision_timeout_s
        if self.agent.is_sync:
            if self._inflight is not None and not self._inflight.done():
                raise RuntimeError("agent still busy with an earlier step")
            fut = asyncio.get_running_loop().run_in_executor(self._agent_pool, self.agent.decide_sync, obs)
            self._inflight = fut
            return await asyncio.wait_for(asyncio.shield(fut), timeout)  # thread keeps running on timeout
        return await asyncio.wait_for(self.agent.decide(obs), timeout)   # async agents are cancelled

    async def _decide(self, obs: dict) -> tuple[Decision, bool, str | None]:
        try:
            d = await self._call_agent(obs)
            if isinstance(d, dict):
                d = Decision.from_dict(d)
            if not isinstance(d, Decision):
                raise TypeError(f"agent returned {type(d).__name__}, expected Decision")
            d.agent = d.agent or self.agent.name
            return d, False, None
        except Exception as e:  # timeout, crash, bad output -> safe fallback
            err = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            log.warning("Agent %s failed (%s); using fallback", self.agent.name, err)
            return self._fallback_decision(obs, err), True, err

    def _fallback_decision(self, obs: dict, err: str) -> Decision:
        d = self.fallback.decide_sync(obs)
        d.reasons.insert(0, f"FALLBACK: {self.agent.name} failed ({err}); safety agent took over")
        d.meta["fallback_error"] = err
        return d

    async def _tick(self, run_id: str | None = None) -> None:
        async with self._lock:
            sim = self.sim
            if sim is None or sim.done or (run_id is not None and run_id != self.run_id):
                return
            obs = sim.observe()
            t0 = time.perf_counter()
            decision, fallback, _ = await self._decide(obs)
            latency = (time.perf_counter() - t0) * 1000
            try:
                tick = sim.apply_decision(decision, latency_ms=latency, fallback=fallback)
            except ValueError as e:  # malformed decision is rejected before any state changes
                decision = self._fallback_decision(obs, str(e))
                tick = sim.apply_decision(decision, latency_ms=latency, fallback=True)
            point = chart_point(tick)
            point["objective"] = round(sim.kpis.summary()["objective"]["total"])
            self.chart.append(point)
            entry = {"step": tick["step"], "time": tick["time"], "label": tick["label"], "agent": tick["agent"],
                     "mode": tick["mode"], "reasons": tick["reasons"], "actions": action_summary(tick),
                     "fallback": tick["fallback"], "notes": tick["notes"]}
            self.decision_log.append(entry)
            self.decision_log = self.decision_log[-500:]
            if sim.done:
                self.state = "finished"
            msg = {
                "type": "tick", "run_id": self.run_id, "status": self.status(),
                "tick": public_tick(tick), "point": point, "log": entry,
                "kpis": sim.kpis.summary(),
                "observation": None if sim.done else sim.observe(),
                "events": sim.event_timeline(),
                "planner": getattr(self.agent, "directive", None),
            }
            this_run, summary = self.run_id, (sim.summary() if sim.done else None)
        # persistence and broadcasting happen outside the lock
        await self._db(self.store.add_tick, this_run, tick)
        if summary is not None:
            await self._db(self.store.finish_run, this_run, summary)
        await self.hub.broadcast(msg)

    # ------------------------------------------------------------------ views
    def status(self) -> dict:
        sim = self.sim
        if sim is None:
            return {"state": self.state}
        step = min(sim.step, sim.total_steps)
        return {
            "state": self.state, "run_id": self.run_id, "scenario": sim.scenario.id,
            "scenario_name": sim.scenario.name, "agent": self.agent.name if self.agent else None,
            "seed": sim.seed, "days": sim.days, "step": step, "total_steps": sim.total_steps,
            "time": hhmm(step) if not sim.done else "24:00", "label": label(step) if not sim.done else "end",
            "speed": self.speed, "tick_interval_s": round(self.tick_interval_s, 3),
        }

    def snapshot(self) -> dict:
        sim = self.sim
        return {
            "type": "snapshot",
            "status": self.status(),
            "catalog": catalog(),
            "scenario": sim.scenario.to_dict() if sim else None,
            "agent": self.agent.info() if self.agent else None,
            "assets": sim.portfolio.specs() if sim else None,
            "observation": sim.observe() if sim and not sim.done else None,
            "chart": self.chart,
            "decisions": self.decision_log[-50:],
            "events": sim.event_timeline() if sim else [],
            "kpis": sim.kpis.summary() if sim else None,
            "planner": getattr(self.agent, "directive", None),
            "instructions": self.instructions[-5:],
        }

    def baseline(self) -> dict:
        """Same day (scenario events, same seed) replayed by the reference agents, for live comparison.
        Cached per run; injected events are not replayed."""
        sim = self.sim
        key = (sim.scenario.id, sim.seed, sim.days)
        if getattr(self, "_baseline_key", None) != key:
            out = {}
            for name in ("naive", "rule_based"):
                b = Simulator(sim.scenario, seed=sim.seed, days=sim.days, keep_history=False)
                agent, curve = create_agent(name), []
                while not b.done:
                    b.apply_decision(agent.decide_sync(b.observe()))
                    curve.append(round(b.kpis.summary()["objective"]["total"]))
                out[name] = {"kpis": b.kpis.summary(), "objective_curve": curve}
            self._baseline_key, self._baseline = key, out
        return {"scenario": key[0], "seed": key[1], "agents": self._baseline}

    async def _broadcast_status(self) -> None:
        await self.hub.broadcast({"type": "status", "status": self.status()})
