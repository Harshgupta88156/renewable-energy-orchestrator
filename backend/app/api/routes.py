"""REST + WebSocket API. Interactive docs: http://localhost:8000/docs"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from ..runtime.session import SimulationSession, catalog
from .schemas import AgentRequest, EventRequest, LabRequest, ResetRequest, SpeedRequest, StepRequest

router = APIRouter(prefix="/api")
ws_router = APIRouter()


def _session(request: Request) -> SimulationSession:
    return request.app.state.session


def _bad(e: Exception) -> HTTPException:
    # KeyError wraps its message in quotes; use the raw argument instead
    msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
    return HTTPException(status_code=400, detail=str(msg))


# ------------------------------------------------------------------ info
@router.get("/health", tags=["info"])
async def health():
    return {"ok": True}


@router.get("/meta", tags=["info"], summary="Scenarios, agents, event catalog, assets, objective weights")
async def meta(request: Request):
    s = _session(request)
    return {**catalog(), "assets": s.sim.portfolio.specs() if s.sim else None}


@router.get("/status", tags=["simulation"])
async def status(request: Request):
    return _session(request).status()


@router.get("/snapshot", tags=["simulation"], summary="Everything a dashboard needs to render (same as first WS message)")
async def snapshot(request: Request):
    return _session(request).snapshot()


@router.get("/observation", tags=["simulation"], summary="Exactly what the agent sees right now")
async def observation(request: Request):
    s = _session(request)
    if not s.sim or s.sim.done:
        raise HTTPException(409, "No active step (simulation finished or not started)")
    return s.sim.observe()


@router.get("/history", tags=["simulation"], summary="Chart points and decision log of the current run")
async def history(request: Request, since: int = 0):
    s = _session(request)
    return {"run_id": s.run_id, "chart": [p for p in s.chart if p["step"] >= since],
            "decisions": [d for d in s.decision_log if d["step"] >= since]}


@router.get("/ticks", tags=["simulation"], summary="Full tick records (decision, applied actions, flows, costs, notes)")
async def ticks(request: Request, since: int = 0, limit: int = 96):
    s = _session(request)
    if not s.sim:
        return []
    out = [t for t in s.sim.history if t["step"] >= since][:limit]
    return [{k: v for k, v in t.items() if k != "kpi_delta"} for t in out]


@router.get("/kpis", tags=["simulation"])
async def kpis(request: Request):
    s = _session(request)
    return s.sim.kpis.summary() if s.sim else {}


# ------------------------------------------------------------------ control
@router.post("/sim/reset", tags=["control"])
async def reset(body: ResetRequest, request: Request):
    try:
        return await _session(request).reset(body.scenario, body.seed, body.agent, body.days, body.speed,
                                             body.autostart)
    except (KeyError, ValueError) as e:
        raise _bad(e)


@router.post("/sim/start", tags=["control"])
async def start(request: Request):
    return await _session(request).start()


@router.post("/sim/pause", tags=["control"])
async def pause(request: Request):
    return await _session(request).pause()


@router.post("/sim/resume", tags=["control"])
async def resume(request: Request):
    return await _session(request).start()


@router.post("/sim/step", tags=["control"], summary="Advance N steps (pauses the clock)")
async def step(request: Request, body: StepRequest | None = None):
    return await _session(request).step((body or StepRequest()).steps)


@router.post("/sim/speed", tags=["control"])
async def speed(body: SpeedRequest, request: Request):
    return await _session(request).set_speed(body.speed)


@router.post("/sim/agent", tags=["control"], summary="Swap the decision maker mid-run")
async def agent(body: AgentRequest, request: Request):
    try:
        return await _session(request).set_agent(body.agent)
    except KeyError as e:
        raise _bad(e)


# ------------------------------------------------------------------ events
@router.get("/events", tags=["events"], summary="Event timeline of the current run")
async def events(request: Request):
    s = _session(request)
    return s.sim.event_timeline() if s.sim else []


@router.post("/events", tags=["events"], summary="Inject an event (dashboard 'chaos buttons')")
async def inject(body: EventRequest, request: Request):
    duration = "default" if body.duration_steps == -1 else body.duration_steps
    try:
        return await _session(request).inject_event(body.type, body.params, body.start_in_steps, duration,
                                                    body.announce_in_steps, body.severity)
    except (KeyError, ValueError, TypeError) as e:
        raise _bad(e)


# ------------------------------------------------------------------ audit log
@router.get("/runs", tags=["audit"])
async def runs(request: Request, limit: int = 50):
    return await asyncio.to_thread(request.app.state.store.list_runs, limit)


@router.get("/runs/{run_id}", tags=["audit"])
async def run_detail(run_id: str, request: Request):
    store = request.app.state.store
    run = await asyncio.to_thread(store.get_run, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    run["events"] = await asyncio.to_thread(store.get_events, run_id)
    return run


@router.get("/runs/{run_id}/ticks", tags=["audit"])
async def run_ticks(run_id: str, request: Request, since: int = 0, limit: int = 96):
    return await asyncio.to_thread(request.app.state.store.get_ticks, run_id, since, limit)


# ------------------------------------------------------------------ scenario lab (F3)
@router.post("/lab/runs", tags=["scenario lab"], summary="Start a Monte Carlo batch comparing agents")
async def lab_submit(body: LabRequest, request: Request):
    try:
        return request.app.state.lab.submit(body.model_dump())
    except (KeyError, ValueError) as e:
        raise _bad(e)


@router.get("/lab/runs", tags=["scenario lab"])
async def lab_list(request: Request):
    return request.app.state.lab.list()


@router.get("/lab/runs/{job_id}", tags=["scenario lab"])
async def lab_get(job_id: str, request: Request):
    job = request.app.state.lab.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# ------------------------------------------------------------------ websocket
@ws_router.websocket("/ws")
async def websocket(ws: WebSocket):
    """Live stream. First message: snapshot. Then: tick | event | status | lab_progress | lab_done | error.
    Clients may also send {"action": "start"|"pause"|"step"|"speed"|"event", ...}."""
    s: SimulationSession = ws.app.state.session
    hub = ws.app.state.hub
    await hub.connect(ws, s.snapshot())
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            try:
                if action == "start":
                    await s.start()
                elif action == "pause":
                    await s.pause()
                elif action == "step":
                    await s.step(int(msg.get("steps", 1)))
                elif action == "speed":
                    await s.set_speed(float(msg["speed"]))
                elif action == "event":
                    await s.inject_event(msg["type"], msg.get("params"), int(msg.get("start_in_steps", 0)))
                elif action == "ping":
                    await ws.send_json({"type": "pong"})
            except Exception as e:  # report, keep the socket open
                await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(ws)
