"""Agentic planner: perceive -> assess risk -> test plans on a digital twin -> decide -> verify -> act.

    every 15 min   the dispatcher executes the current plan (fast, exact, always safe)
    every ~2 h,    1. PERCEIVE  compact situation brief: state, forecast bands, bulletins, operator notes
    on new events  2. SIMULATE  the digital twin tests 9 candidate plans in an expected and a stress world
    or operator    3. REASON    the LLM (free: Groq / Gemini / OpenRouter / local Ollama) judges the risk,
    instructions                 picks a plan from the twin's table, may ask the twin to test its own
                                 variants (tool call), and writes the explanation for the operator
                   4. VERIFY    guardrails: the LLM may only RAISE risk above the detector, and its plan
                                 must score within tolerance of the twin's best, else the best is used
                   5. ACT       the plan configures the dispatcher; physics guard + rule fallback remain

Token budget: one call per review (a second only if it asks the twin), ~1.5k tokens, about
12 reviews per simulated day, all within free-tier limits. If the model is unavailable or over
budget, steps 1, 2, 4, 5 still run ("auto" mode), so the agent never stops working.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from ..sim.decision import Decision, MaintenanceCommand
from .base import Agent
from .llm_providers import LLMClient, RateLimited, make_client, tool_calls
from .rule_based import RuleBasedAgent
from .twin import DEFAULT_PLANS, STRESS_WEIGHT, Plan, Twin, compact

REVIEW_STEPS = int(os.getenv("REO_LLM_REVIEW_STEPS", "8"))     # routine review every 2 h
MAX_CALLS_PER_RUN = int(os.getenv("REO_LLM_MAX_CALLS", "40"))  # hard cap per run
COOLDOWN_STEPS = 4
LEVELS = ["low", "elevated", "high", "critical"]
TOLERANCE = (0.002, 20_000)  # plan may cost up to 0.2 % (min ₹20k) more than the twin's best

SYSTEM_PROMPT = """You are the strategy layer of an autonomous renewable-energy orchestrator for an Indian \
regional grid (5 solar farms, 3 wind farms, 2 batteries, 2 tie-lines, 5 large consumers). \
A dispatcher executes every 15-minute step; you choose its PLAN every ~2 hours.
A plan = reserve_soc (battery floor 0.15-0.80 kept for emergencies), arbitrage (conservative|normal|aggressive \
price trading) and lookahead_h (6|12 h risk look-ahead).
A digital twin has already simulated candidate plans over the next 12 h in an EXPECTED world (P50) and a \
STRESS world (low renewables, high demand and price). Costs are in ₹ lakh, lower is better; unserved energy \
means blackouts. Ranking = blend of expected and stress cost; the stress share grows with risk_level \
(low 20%, elevated 35%, high 50%, critical 65%).
How to judge risk:
- Risk means demand that the grid import limit plus batteries may NOT cover (see detector shortfall MWh, \
bulletins, offline assets, operator notes). Low solar at night or simply buying from the grid is normal, not risk.
- The detector's level is a floor. Raise it only for a concrete reason (a bulletin, an operator note, a band \
that looks worse than the detector says).
- During an active event the reserve exists to be used: do not raise reserve_soc above the batteries' current charge.
How to choose: normally take the best plan_id in twin_results for your risk level. If you pick another, say \
why. Optionally call evaluate_plans once to test a variant; always finish with submit_directive.
Reasons (2-3, shown to the grid operator): cite numbers from the brief only (times, MW, MWh, ₹ lakh), and \
compare with an alternative, e.g. "R50 costs ₹0.4 L more on a normal day but ₹3 L less if the storm \
hits harder". Never invent capacities. Always answer with a tool call, never plain text."""

# Schemas are kept loose on purpose: strict enums/limits make free models fail server-side
# validation. Every value is checked and clamped in code instead.
TOOLS = [
    {"name": "evaluate_plans", "description": "Simulate up to 4 extra plans on the digital twin.",
     "parameters": {"type": "object", "properties": {"plans": {"type": "array", "items": {
         "type": "object", "properties": {
             "reserve_soc": {"type": "number", "description": "0.15 to 0.80"},
             "arbitrage": {"type": "string", "description": "conservative | normal | aggressive"},
             "lookahead_h": {"type": "integer", "description": "6 or 12"}}}}},
         "required": ["plans"]}},
    {"name": "submit_directive", "description": "Commit the plan for the next hours. Always end with this call.",
     "parameters": {"type": "object", "properties": {
         "risk_level": {"type": "string", "description": "low | elevated | high | critical"},
         "plan_id": {"type": "string", "description": "a plan_id from twin_results"},
         "assessment": {"type": "string", "description": "one sentence: what matters in the next 12 h"},
         "reasons": {"type": "array", "items": {"type": "string"}, "description": "2-3 short reasons"},
         "valid_steps": {"type": "integer", "description": "4 to 16"}},
         "required": ["risk_level", "plan_id", "assessment", "reasons"]}},
]


# ---------------------------------------------------------------------------- perception
def detect_risk(obs: dict) -> tuple[str, list[str]]:
    """Deterministic risk floor from forecast bands, bulletins and asset status."""
    fc, dt, cur = obs["forecast"], obs["dt_hours"], obs["current"]
    h = min(48, fc["horizon_steps"])
    ren10 = np.array(fc["solar_total_mw"]["p10"][:h]) + np.array(fc["wind_total_mw"]["p10"][:h])
    gap = np.maximum(0.0, np.array(fc["demand_total_mw"]["p90"][:h]) - ren10 - np.array(fc["import_limit_mw"][:h]))
    short = float(gap.sum() * dt)
    level, why = 0, []
    if short > 5:
        level = 1 if short < 80 else 2
        why.append(f"{short:.0f} MWh pessimistic shortfall beyond grid limits in 12 h "
                   f"(worst {fc['times'][int(np.argmax(gap))]})")
    for a in obs["alerts"]:
        soon = a.get("status") == "active" or (a.get("starts_in_steps") or 999) <= 48
        if soon and a.get("severity") in ("warning", "critical"):
            lv = 3 if a["severity"] == "critical" and a.get("status") == "active" else 2
            if lv > level:
                level = lv
            why.append(f"{a['title']} ({a.get('status')}, {a.get('start')})")
    if any(b["status"] != "online" for b in cur["batteries"].values()):
        level = max(level, 1)
        why.append("a battery is offline")
    return LEVELS[level], why[:3]


def build_brief(obs: dict, rows: list[dict], floor: str, floor_why: list[str], notes: list[str],
                current_plan: str | None) -> dict:
    cur, fc, dt = obs["current"], obs["forecast"], obs["dt_hours"]
    h = min(48, fc["horizon_steps"])
    price = np.array(fc["price"]["p50"][:h])
    return {
        "now": obs["label"], "steps_left_in_run": obs["steps_remaining"],
        "state": {"renewable_mw": round(cur["renewable_total_mw"]), "demand_mw": round(cur["demand_total_mw"]),
                  "net_mw": round(cur["net_position_mw"]), "import_limit_mw": round(cur["grid"]["import_limit_mw"]),
                  "price_rs_mwh": round(cur["market"]["price"])},
        "batteries": {k: f"{b['status']} soc {b['soc']:.0%} of {b['effective_energy_mwh']:.0f} MWh"
                      for k, b in cur["batteries"].items()},
        "next_12h": {
            "renewable_p10_to_p90_mw": [round(min(fc["solar_total_mw"]["p10"][i] + fc["wind_total_mw"]["p10"][i]
                                                  for i in range(h))),
                                        round(max(fc["solar_total_mw"]["p90"][i] + fc["wind_total_mw"]["p90"][i]
                                                  for i in range(h)))],
            "demand_peak_p90_mw": round(max(fc["demand_total_mw"]["p90"][:h])),
            "min_import_limit_mw": round(min(fc["import_limit_mw"][:h])),
            "price_p50_min_max": [round(price.min()), round(price.max())],
            "price_peak_at": fc["times"][int(np.argmax(price))],
        },
        "bulletins": [f"[{a.get('severity')}/{a.get('status')}] {a.get('bulletin') or a.get('title')}"
                      for a in obs["alerts"][:6]],
        "operator_notes": notes[-3:],
        "detector": {"risk_floor": floor, "why": floor_why},
        "current_plan": current_plan,
        "twin_results": compact(rows, 6),
    }


# ---------------------------------------------------------------------------- the agent
class LLMPlannerAgent(Agent):
    name = "llm_planner"
    label = "Agentic planner (LLM + digital twin)"
    description = ("Every ~2 h or on new events: reads bulletins and operator notes, tests 9+ plans on a "
                   "digital twin (expected and stress case), lets an LLM judge the risk and pick the plan, "
                   "verifies it against guardrails, then the dispatcher executes it. Runs on free LLMs "
                   "(Groq, Gemini, OpenRouter, local Ollama) or fully offline.")

    def __init__(self, client: LLMClient | None | str = "env"):
        self.client = make_client() if client == "env" else client
        self.inner = RuleBasedAgent()
        self.instructions: list[str] = []
        prov = self.client.name if self.client else "offline"
        self.label = f"Agentic planner ({prov} + digital twin)"
        self.reset()

    def reset(self) -> None:
        self.plan = Plan()
        self.directive: dict | None = None
        self.valid_until = -1
        self.seen_events: set[str] = set()
        self.calls = self.errors = self.skipped = self.overrides = 0
        self.tokens = {"in": 0, "out": 0}
        self.cooldown_until = -1
        self.last_error: str | None = None
        self._notes_seen = len(self.instructions)
        self.plan.configure(self.inner)

    def instruct(self, text: str) -> dict:
        """Operator guidance in plain language ("cyclone warning for tonight, keep batteries full")."""
        text = str(text).strip()[:300]
        if text:
            self.instructions.append(text)
        return {"instructions": self.instructions[-5:]}

    # ---- when to plan
    def _trigger(self, obs: dict) -> str | None:
        ids = {a["id"] for a in obs["alerts"]}
        if self.directive is None:
            return "start"
        if ids - self.seen_events:
            return "new event"
        if len(self.instructions) > self._notes_seen:
            return "operator note"
        if obs["step"] >= self.valid_until:
            return "scheduled review"
        return None

    # ---- the LLM step (tool-calling loop, at most 2 calls)
    def _ask_llm(self, obs: dict, twin: Twin, rows: list[dict], brief: dict) -> dict | None:
        if self.client is None or self.calls >= MAX_CALLS_PER_RUN or obs["step"] < self.cooldown_until:
            return None
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(brief, separators=(",", ":"), ensure_ascii=False)}]
        force = None
        for round_ in range(2):
            self.calls += 1
            try:
                msg, usage = self.client.chat(msgs, TOOLS, force_tool=force)
            except RateLimited:
                self.calls -= 1
                self.skipped += 1
                return None
            except Exception as e:  # network, auth, bad output: retry once forcing the directive
                self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
                if round_ == 0 and "HTTP 400" in str(e):
                    force = "submit_directive"
                    continue
                self.errors += 1
                self.cooldown_until = obs["step"] + COOLDOWN_STEPS
                return None
            self.tokens["in"] += usage["in"]
            self.tokens["out"] += usage["out"]
            calls = tool_calls(msg)
            sub = next((a for n, a, _ in calls if n == "submit_directive"), None)
            if sub is not None:
                self.last_error = None
                return sub
            force = "submit_directive"
            ev = next(((a, cid) for n, a, cid in calls if n == "evaluate_plans"), None)
            if ev is None:
                continue  # answered in prose: ask again, this time forcing the directive
            extra = [Plan.from_any(p) for p in (ev[0].get("plans") or [])[:4]]
            new = twin.evaluate(extra) if extra else []
            rows.extend(r for r in new if r["plan_id"] not in {x["plan_id"] for x in rows})
            msgs += [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls")},
                     {"role": "tool", "tool_call_id": ev[1], "content": json.dumps(compact(new, 4))}]
        self.errors += 1
        self.last_error = "model did not submit a directive"
        return None

    # ---- perceive -> simulate -> reason -> verify
    def _plan(self, obs: dict, trigger: str) -> None:
        t0 = time.perf_counter()
        self.seen_events |= {a["id"] for a in obs["alerts"]}
        self._notes_seen = len(self.instructions)
        floor, floor_why = detect_risk(obs)
        twin = Twin(obs)
        rows = twin.evaluate(list(dict.fromkeys([*DEFAULT_PLANS, self.plan])))
        brief = build_brief(obs, _rank(rows, floor), floor, floor_why, self.instructions, self.plan.id)
        raw = self._ask_llm(obs, twin, rows, brief)

        source = "llm" if raw else "auto"
        risk = raw.get("risk_level") if raw and raw.get("risk_level") in LEVELS else floor
        if LEVELS.index(risk) < LEVELS.index(floor):
            risk = floor                                  # guardrail 1: the model may raise risk, never lower it
        ranked = _rank(rows, risk)
        best = ranked[0]
        chosen = next((r for r in ranked if raw and r["plan_id"] == raw.get("plan_id")), best)
        note = None
        tol = max(TOLERANCE[0] * abs(best["score"]), TOLERANCE[1])
        if chosen["score"] - best["score"] > tol:       # guardrail 2: the twin must agree within tolerance
            self.overrides += 1
            note = (f"Twin check: {chosen['plan_id']} would cost ₹{(chosen['score'] - best['score']) / 1e5:.2f} L "
                    f"more than {best['plan_id']}; using {best['plan_id']}")
            chosen = best
        self.plan = Plan(chosen["reserve_soc"], chosen["arbitrage"], chosen["lookahead_h"])
        self.plan.configure(self.inner)

        if raw:
            reasons = [str(r)[:220] for r in (raw.get("reasons") or []) if isinstance(r, str)][:3]
            assessment = str(raw.get("assessment") or "")[:240]
            try:
                valid = int(raw.get("valid_steps") or REVIEW_STEPS)
            except (TypeError, ValueError):
                valid = REVIEW_STEPS
        else:
            reasons, assessment, valid = floor_why[:2], "", REVIEW_STEPS
        default = next((r for r in ranked if r["plan_id"] == DEFAULT_PLANS[0].id), None)
        saving = (default["score"] - chosen["score"]) if default else 0.0
        if not assessment:
            assessment = (f"{len(rows)} plans tested on the twin; {chosen['plan_id']} is best at risk {risk}"
                          + (f", ₹{saving / 1e5:.2f} L better than the default plan" if saving > 1000 else ""))
        if note:
            reasons.append(note)
        self.directive = {
            "step": obs["step"], "trigger": trigger, "source": source, "risk_level": risk, "risk_floor": floor,
            "plan": self.plan.id, "assessment": assessment, "reasons": reasons,
            "saving_vs_default_rs": round(saving), "evaluated": compact(ranked, 5),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }
        self.valid_until = obs["step"] + int(np.clip(valid, 4, 16))

    # ---- one 15-minute step
    def decide_sync(self, obs: dict) -> Decision:
        trigger = self._trigger(obs)
        if trigger:
            self._plan(obs, trigger)
        dec = self.inner.decide_sync(obs)
        d = self.directive
        if trigger and d:
            tag = f"[{'LLM' if d['source'] == 'llm' else 'Planner'} · {d['trigger']} · risk {d['risk_level']} · plan {d['plan']}]"
            dec.reasons = [f"{tag} {d['assessment']}"] + [f"[{d['source'].upper()}] {r}" for r in d["reasons"]] + dec.reasons
        if d and d["risk_level"] in ("high", "critical") and dec.mode == "normal":
            dec.mode = "reliability_prep"
        dec.agent = self.name
        dec.meta["planner"] = {
            "provider": self.client.name if self.client else "offline", "model": self.client.model if self.client else None,
            "source": d["source"] if d else None, "risk_level": d["risk_level"] if d else None,
            "plan": self.plan.id, "calls": self.calls, "tokens": dict(self.tokens), "errors": self.errors,
            "skipped_budget": self.skipped, "twin_overrides": self.overrides, "error": self.last_error,
            "reviewed_now": bool(trigger),
        }
        if trigger and d:
            dec.meta["planner"]["evaluated"] = d["evaluated"]
            dec.meta["planner"]["saving_vs_default_rs"] = d["saving_vs_default_rs"]
        return dec


def _rank(rows: list[dict], risk: str) -> list[dict]:
    w = STRESS_WEIGHT[risk]
    out = [dict(r, score=(1 - w) * r["cost_expected"] + w * r["cost_stress"]) for r in rows]
    return sorted(out, key=lambda r: r["score"])
