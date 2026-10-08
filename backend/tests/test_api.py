"""API + WebSocket contract, and the safety fallback."""
import pytest
from fastapi.testclient import TestClient

from app.agents import AGENT_CLASSES, Agent, create_agent
from app.main import app


class BrokenAgent(Agent):
    name = "broken"
    label = "Broken (test)"

    def decide_sync(self, obs):
        raise RuntimeError("LLM returned garbage")


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_meta_lists_everything(client):
    m = client.get("/api/meta").json()
    assert {s["id"] for s in m["scenarios"]} >= {"storm_alert", "chaos_monkey"}
    assert {"naive", "rule_based"} <= {a["name"] for a in m["agents"]}
    assert "storm" in m["events"] and m["assets"]["totals"]["solar_capacity_mw"] == 300


def test_reset_step_inject_and_observe(client):
    r = client.post("/api/sim/reset", json={"scenario": "normal_day", "agent": "rule_based", "seed": 3})
    assert r.status_code == 200 and r.json()["step"] == 0
    assert client.post("/api/sim/step", json={"steps": 4}).json()["step"] == 4
    ev = client.post("/api/events", json={"type": "cloud_cover", "params": {"drop": 0.9}}).json()
    assert ev["status"] == "active"
    obs = client.get("/api/observation").json()
    assert ev["id"] in {a["id"] for a in obs["alerts"]}
    assert len(obs["forecast"]["price"]["p50"]) == 96
    hist = client.get("/api/history").json()
    assert len(hist["chart"]) == 4 and hist["decisions"][0]["actions"]


def test_bad_requests_return_400(client):
    assert client.post("/api/sim/reset", json={"scenario": "nope"}).status_code == 400
    assert client.post("/api/events", json={"type": "nope"}).status_code == 400
    r = client.post("/api/events", json={"type": "asset_fault", "params": {"asset_id": "XX"}})
    assert r.status_code == 400 and "XX" in r.json()["detail"]


def test_websocket_sends_snapshot_then_ticks(client):
    client.post("/api/sim/reset", json={"scenario": "storm_alert"})
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot" and first["observation"]["step"] == 0
        ws.send_json({"action": "step", "steps": 1})
        msg = ws.receive_json()
        while msg["type"] != "tick":
            msg = ws.receive_json()
        assert msg["tick"]["step"] == 0 and "kpi_delta" not in msg["tick"]
        assert msg["observation"]["step"] == 1


def test_broken_agent_falls_back_safely(client):
    AGENT_CLASSES["broken"] = BrokenAgent
    try:
        client.post("/api/sim/reset", json={"scenario": "normal_day", "agent": "broken"})
        client.post("/api/sim/step", json={"steps": 2})
        log = client.get("/api/history").json()["decisions"]
        assert all(d["fallback"] for d in log)
        assert "FALLBACK" in log[0]["reasons"][0]
        assert client.get("/api/kpis").json()["agent"]["fallbacks"] == 2
    finally:
        AGENT_CLASSES.pop("broken", None)


class SlowAgent(Agent):
    name = "slow"
    label = "Slow (test)"

    def decide_sync(self, obs):
        import time
        time.sleep(0.6)
        return create_agent("naive").decide_sync(obs)


def test_bad_event_via_api_does_not_break_the_run(client):
    client.post("/api/sim/reset", json={"scenario": "normal_day"})
    r = client.post("/api/events", json={"type": "storm", "params": {"solar_drop": "heavy"}})
    assert r.status_code == 400
    assert client.get("/api/observation").status_code == 200
    assert client.post("/api/sim/step", json={"steps": 2}).json()["step"] == 2


def test_slow_agent_times_out_and_never_blocks(client):
    import dataclasses
    session = client.app.state.session
    old = session.settings
    session.settings = dataclasses.replace(old, decision_timeout_s=0.2)
    AGENT_CLASSES["slow"] = SlowAgent
    try:
        client.post("/api/sim/reset", json={"scenario": "normal_day", "agent": "slow"})
        client.post("/api/sim/step", json={"steps": 3})
        log = client.get("/api/history").json()["decisions"]
        assert len(log) == 3 and all(d["fallback"] for d in log)
        assert any("busy" in d["reasons"][0] for d in log[1:])
    finally:
        session.settings = old
        AGENT_CLASSES.pop("slow", None)


def test_baseline_and_dashboard(client):
    client.post("/api/sim/reset", json={"scenario": "normal_day", "seed": 3})
    b = client.get("/api/baseline").json()
    assert set(b["agents"]) == {"naive", "rule_based"}
    assert len(b["agents"]["rule_based"]["objective_curve"]) == 96
    snap = client.get("/api/snapshot").json()
    assert "planner" in snap and "instructions" in snap
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (200, 307)
