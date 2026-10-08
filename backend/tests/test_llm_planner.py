import json

import pytest

from app.agents import list_agents
from app.agents.llm_planner import LLMPlannerAgent, build_brief, detect_risk
from app.agents.llm_providers import LLMClient, RateLimited, tool_calls
from app.agents.twin import DEFAULT_PLANS, Plan, Twin
from app.sim.engine import Simulator
from app.sim.scenarios import SCENARIOS


def run(agent, scenario="storm_alert", seed=7, steps=96):
    sim = Simulator(SCENARIOS[scenario], seed=seed)
    ds = []
    for _ in range(steps):
        if sim.done:
            break
        d = agent.decide_sync(sim.observe())
        ds.append(d)
        sim.apply_decision(d)
    return sim, ds


def fake(reply_fn):
    """LLMClient whose HTTP call is replaced: reply_fn(body) -> assistant message dict."""
    seen = []

    def post(url, body, headers, timeout):
        seen.append((url, body, headers))
        return {"choices": [{"message": reply_fn(body)}], "usage": {"prompt_tokens": 1200, "completion_tokens": 150}}
    c = LLMClient("groq", "https://x/v1", "k", "m", tpm=10**9, post=post)
    return c, seen


def submit(args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "submit_directive", "arguments": json.dumps(args)}}]}


def test_registered():
    assert "llm_planner" in {a["name"] for a in list_agents()}


def test_offline_planner_is_safe_and_not_worse():
    from app.agents.rule_based import RuleBasedAgent
    a = LLMPlannerAgent(client=None)
    sim, ds = run(a)
    base, _ = run(RuleBasedAgent())
    assert sim.kpis.summary()["energy"]["unserved_mwh"] <= base.kpis.summary()["energy"]["unserved_mwh"]
    assert sim.kpis.summary()["objective"]["total"] <= base.kpis.summary()["objective"]["total"] + 1e4
    assert any(r.startswith("[Planner") for d in ds for r in d.reasons)
    assert sum(d.meta["planner"]["reviewed_now"] for d in ds) <= 20


def test_twin_uses_only_observation_and_ranks():
    sim = Simulator(SCENARIOS["storm_alert"], seed=7)
    for _ in range(50):
        sim.apply_decision(None)
    obs = sim.observe()
    rows = Twin(obs).evaluate(DEFAULT_PLANS, "high")
    assert len(rows) == len(DEFAULT_PLANS)
    assert rows == sorted(rows, key=lambda r: r["score"])
    assert all(r["cost_stress"] >= r["cost_expected"] - 1e5 for r in rows)  # stress world is harder


def test_brief_is_small():
    sim = Simulator(SCENARIOS["chaos_monkey"], seed=3)
    obs = sim.observe()
    rows = Twin(obs).evaluate(DEFAULT_PLANS)
    floor, why = detect_risk(obs)
    assert len(json.dumps(build_brief(obs, rows, floor, why, ["note"], "R15-norm-6h"))) < 3500  # < ~1k tokens


def test_llm_choice_used_with_tokens_counted():
    c, seen = fake(lambda body: submit({"risk_level": "high", "plan_id": "R50-norm-12h", "assessment": "storm 17:00",
                                        "reasons": ["Storm from 17:00: keep 50% in batteries"], "valid_steps": 8}))
    a = LLMPlannerAgent(client=c)
    _, ds = run(a, steps=60)
    assert a.calls >= 1 and a.tokens["in"] >= 1200
    assert seen[0][1]["tools"][1]["function"]["name"] == "submit_directive"
    assert seen[0][2]["Authorization"] == "Bearer k"
    assert any("[LLM" in r for d in ds for r in d.reasons)


def test_guardrails_risk_floor_and_twin_override():
    c, _ = fake(lambda body: submit({"risk_level": "low", "plan_id": "R70-norm-12h", "assessment": "x", "reasons": []}))
    a = LLMPlannerAgent(client=c)
    sim = Simulator(SCENARIOS["storm_alert"], seed=7)
    for _ in range(56):  # storm announced, starts in a few hours
        sim.apply_decision(None)
    a.decide_sync(sim.observe())
    d = a.directive
    floor = d["risk_floor"]
    assert ["low", "elevated", "high", "critical"].index(d["risk_level"]) >= ["low", "elevated", "high", "critical"].index(floor)
    best = d["evaluated"][0]["plan_id"]
    assert a.plan.id == best or a.overrides == 0


def test_tool_loop_evaluate_then_submit():
    state = {"n": 0}

    def reply(body):
        state["n"] += 1
        if state["n"] == 1:
            return {"role": "assistant", "content": None, "tool_calls": [{"id": "e1", "type": "function", "function": {
                "name": "evaluate_plans", "arguments": json.dumps({"plans": [{"reserve_soc": 0.4, "arbitrage": "aggressive",
                                                                              "lookahead_h": 12}]})}}]}
        assert body["messages"][-1]["role"] == "tool"
        assert body["tool_choice"]["function"]["name"] == "submit_directive"
        return submit({"risk_level": "elevated", "plan_id": "R40-aggr-12h", "assessment": "ok", "reasons": ["r"]})
    c, _ = fake(reply)
    a = LLMPlannerAgent(client=c)
    run(a, steps=1)
    assert state["n"] == 2 and any(r["plan_id"] == "R40-aggr-12h" for r in a.directive["evaluated"]) or a.overrides == 1


def test_failures_and_budget_fall_back_to_auto():
    def boom(*a, **k):
        raise TimeoutError("slow")
    c = LLMClient("groq", "https://x/v1", "k", "m", tpm=10**9, post=boom)
    a = LLMPlannerAgent(client=c)
    sim, ds = run(a, steps=40)
    assert a.errors >= 1 and ds[-1].meta["planner"]["error"].startswith("TimeoutError")
    assert all(d.meta["planner"]["plan"] for d in ds)
    c2 = LLMClient("groq", "https://x/v1", "k", "m", tpm=10)  # budget too small: never calls the network
    with pytest.raises(RateLimited):
        c2.chat([], [])
    a2 = LLMPlannerAgent(client=c2)
    run(a2, steps=10)
    assert a2.skipped >= 1 and a2.errors == 0


def test_tool_call_parsing_is_tolerant():
    assert tool_calls({"content": 'Sure! {"risk_level": "low", "plan_id": "R15-norm-6h"}'})[0][1]["plan_id"] == "R15-norm-6h"
    assert tool_calls({"tool_calls": [{"function": {"name": "submit_directive", "arguments": "not json"}}]})[0][1] == {}
    assert Plan.from_any({"reserve_soc": 5, "arbitrage": "yolo", "lookahead_h": "x"}) == Plan(0.8, "normal", 6)


def test_operator_instruction_triggers_review():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        _instruction(client)


def _instruction(client):
    r = client.post("/api/sim/reset", json={"scenario": "storm_alert", "agent": "llm_planner", "seed": 7})
    assert r.status_code == 200
    r = client.post("/api/agent/instruction", json={"text": "Cyclone warning tonight, keep batteries full"})
    assert r.status_code == 200 and r.json()["instructions"][-1].startswith("Cyclone")


def test_retired_model_switches_automatically():
    import io
    import urllib.error
    calls = []

    def post(url, body, headers, timeout):
        calls.append(body["model"])
        if body["model"] == "old-model":
            raise urllib.error.HTTPError(url, 404, "model_not_found", {}, io.BytesIO(b"{}"))
        return {"choices": [{"message": submit({"risk_level": "low", "plan_id": "R15-norm-6h",
                                                  "assessment": "a", "reasons": []})}], "usage": {}}
    c = LLMClient("groq", "https://x/v1", "k", "old-model", tpm=10**9, post=post,
                  get=lambda url, h, t: {"data": [{"id": "whisper"}, {"id": "openai/gpt-oss-20b"}]})
    msg, _ = c.chat([], [])
    assert c.model == "openai/gpt-oss-20b" and calls == ["old-model", "openai/gpt-oss-20b"]


def test_bad_tool_call_retries_once_forcing_the_directive():
    import io
    import urllib.error
    bodies = []

    def post(url, body, headers, timeout):
        bodies.append(body)
        if len(bodies) == 1:
            raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b'{"error":"Failed to parse tool call arguments as JSON"}'))
        return {"choices": [{"message": submit({"risk_level": "high", "plan_id": "R50-norm-12h",
                                                  "assessment": "a", "reasons": ["b"]})}], "usage": {}}
    a = LLMPlannerAgent(client=LLMClient("groq", "https://x/v1", "k", "openai/gpt-oss-120b", tpm=10**9, post=post))
    run(a, steps=1)
    assert a.directive["source"] == "llm" and a.errors == 0
    assert bodies[0]["tool_choice"] == "auto" and bodies[0]["reasoning_effort"] == "low"
    assert bodies[1]["tool_choice"]["function"]["name"] == "submit_directive"
