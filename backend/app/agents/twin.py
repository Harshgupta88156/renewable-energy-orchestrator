"""Digital twin: a what-if tool the planner uses to test strategies BEFORE acting.

It is built only from the observation (what an operator can see): current state + the published
forecast. It never touches the simulator's hidden truth, so there is no look-ahead cheating.

For each candidate plan it rolls the real dispatcher forward over the next hours in two worlds:
    expected   P50 renewables, P50 demand, P50 price
    stress     P10 renewables, P90 demand, P90 price   (the bad-but-plausible case)
and returns the cost of each (energy + carbon + demand-side actions + battery wear + unserved
energy at VoLL), crediting energy left in the battery at the end so plans can't "cheat" by
emptying it. The planner ranks plans by a risk-weighted blend of the two.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from ..sim.decision import Decision
from .rule_based import CHEAP_Q, EXPENSIVE_Q, NORMAL_RESERVE, RISK_LOOKAHEAD, RuleBasedAgent

HORIZON = 48  # steps (12 h)
ARBITRAGE = {"conservative": (0.25, 0.80), "normal": (CHEAP_Q, EXPENSIVE_Q), "aggressive": (0.45, 0.60)}
STRESS_WEIGHT = {"low": 0.2, "elevated": 0.35, "high": 0.5, "critical": 0.65}


@dataclass(frozen=True)
class Plan:
    reserve_soc: float = NORMAL_RESERVE
    arbitrage: str = "normal"
    lookahead_h: int = RISK_LOOKAHEAD // 4

    @property
    def id(self) -> str:
        return f"R{round(self.reserve_soc * 100)}-{self.arbitrage[:4]}-{self.lookahead_h}h"

    @classmethod
    def from_any(cls, d) -> "Plan":
        if isinstance(d, Plan):
            return d
        d = d if isinstance(d, dict) else {}
        try:
            r = float(d.get("reserve_soc", NORMAL_RESERVE))
        except (TypeError, ValueError):
            r = NORMAL_RESERVE
        r = float(np.clip(r if r == r else NORMAL_RESERVE, NORMAL_RESERVE, 0.8))
        arb = d.get("arbitrage") if d.get("arbitrage") in ARBITRAGE else "normal"
        try:
            la = int(d.get("lookahead_h", 6))
        except (TypeError, ValueError):
            la = 6
        return cls(round(r, 2), arb, 12 if la >= 9 else 6)

    def configure(self, agent: RuleBasedAgent) -> RuleBasedAgent:
        agent.reserve_floor = self.reserve_soc
        agent.cheap_q, agent.dear_q = ARBITRAGE[self.arbitrage]
        agent.risk_lookahead = self.lookahead_h * 4
        return agent


DEFAULT_PLANS = [Plan(r, a, la) for r, a, la in (
    (0.15, "normal", 6), (0.15, "aggressive", 6), (0.15, "conservative", 6), (0.15, "normal", 12),
    (0.30, "normal", 6), (0.30, "aggressive", 12), (0.50, "normal", 12), (0.50, "conservative", 12),
    (0.70, "normal", 12))]


def _shift(x, k: int, H: int):
    """Shift a forecast structure forward by k steps (pad with the last value)."""
    if isinstance(x, dict):
        return {key: _shift(v, k, H) for key, v in x.items()}
    if isinstance(x, list) and len(x) == H and H > 0:
        return x[k:] + [x[-1]] * k
    return x


class Twin:
    def __init__(self, obs: dict, horizon: int = HORIZON):
        self.obs = obs
        self.dt = obs["dt_hours"]
        fc = obs["forecast"]
        self.H = max(1, min(horizon, fc["horizon_steps"] + 1, obs["steps_remaining"]))
        cur = obs["current"]
        mk = cur["market"]
        self.sell_ratio = mk["sell_price"] / max(mk["price"], 1.0)
        self.dem_now = max(cur["demand_total_mw"], 1e-6)
        ef = np.array(fc["price"]["p50"]) + np.array(fc["grid_ef"]) * np.array(fc["carbon_price"])
        self.end_value = float(np.median(ef[: self.H])) if len(ef) else mk["price"]
        self._fc_cache: dict[int, dict] = {}

    # ---- the world at rollout step j (j=0 is "now", j>=1 comes from forecast index j-1)
    def _world(self, j: int, case: str) -> dict:
        cur = self.obs["current"]
        if j == 0:
            mk = cur["market"]
            return {"solar": cur["solar_total_mw"], "wind": cur["wind_total_mw"], "demand": cur["demand_total_mw"],
                    "price": mk["price"], "grid_ef": mk["grid_ef"], "carbon": mk["carbon_price"],
                    "dr": mk["dr_incentive"], "imp": cur["grid"]["import_limit_mw"],
                    "exp": cur["grid"]["export_limit_mw"],
                    "bat": {k: b["status"] == "online" for k, b in cur["batteries"].items()}}
        fc, i = self.obs["forecast"], j - 1
        ren_q, dem_q, pr_q = ("p50", "p50", "p50") if case == "expected" else ("p10", "p90", "p90")
        return {"solar": fc["solar_total_mw"][ren_q][i], "wind": fc["wind_total_mw"][ren_q][i],
                "demand": fc["demand_total_mw"][dem_q][i], "price": fc["price"][pr_q][i],
                "grid_ef": fc["grid_ef"][i], "carbon": fc["carbon_price"][i], "dr": fc["dr_incentive"][i],
                "imp": fc["import_limit_mw"][i], "exp": fc["export_limit_mw"][i],
                "bat": {k: v[i] > 0.5 for k, v in fc["battery_available"].items()}}

    def _forecast(self, j: int) -> dict:
        if j not in self._fc_cache:
            fc = self.obs["forecast"]
            f = _shift(fc, j, fc["horizon_steps"]) if j else fc
            if j:
                f = dict(f, start_step=fc["start_step"] + j)
            self._fc_cache[j] = f
        return self._fc_cache[j]

    def _obs(self, j: int, w: dict, bats: dict) -> dict:
        o, cur = self.obs, self.obs["current"]
        if j == 0:
            c = dict(cur, batteries=bats)
        else:
            f = w["demand"] / self.dem_now
            scale = ("demand_mw", "critical_mw", "dr_available_mw", "shiftable_mw", "noncritical_mw")
            demand = {cid: {**c, **{k: c[k] * f for k in scale if k in c}} for cid, c in cur["demand"].items()}
            ren = w["solar"] + w["wind"]
            c = dict(cur, batteries=bats, demand=demand, solar_total_mw=w["solar"], wind_total_mw=w["wind"],
                     renewable_total_mw=ren, demand_total_mw=w["demand"], net_position_mw=ren - w["demand"],
                     grid=dict(cur["grid"], import_limit_mw=w["imp"], export_limit_mw=w["exp"], frequency_hz=50.0),
                     market=dict(cur["market"], price=w["price"], sell_price=w["price"] * self.sell_ratio,
                                 grid_ef=w["grid_ef"], carbon_price=w["carbon"], dr_incentive=w["dr"]))
        return dict(o, step=o["step"] + j, steps_remaining=o["steps_remaining"] - j, current=c,
                    forecast=self._forecast(j), alerts=[], maintenance=[], pending_shifted_load=[])

    # ---- one rollout
    def rollout(self, plan: Plan, case: str) -> dict:
        agent = plan.configure(RuleBasedAgent())
        mk, dt = self.obs["current"]["market"], self.dt
        bats = copy.deepcopy(self.obs["current"]["batteries"])
        pending = np.zeros(self.H + 64)
        cost = unserved = carbon_t = dr_mwh = 0.0
        for j in range(self.H):
            w = self._world(j, case)
            for bid, b in bats.items():
                b["status"] = "online" if w["bat"].get(bid, True) and b["status"] != "faulted" else "offline"
            d: Decision = agent.decide_sync(self._obs(j, w, bats))
            ren = w["solar"] + w["wind"] - min(sum(d.curtail_mw.values()), w["solar"] + w["wind"])
            dr = sum(d.demand_response_mw.values())
            cut = sum(d.reduce_noncritical_mw.values())
            sh = 0.0
            for s in d.shift_load:
                sh += s.mw
                pending[j + max(1, s.delay_steps)] += s.mw
            charge = discharge = 0.0
            for bid, cmd in d.battery.items():
                b = bats[bid]
                if b["status"] != "online":
                    continue
                E = b["effective_energy_mwh"]
                if cmd.action == "charge":
                    p = min(cmd.mw, b["max_charge_mw"], max(0.0, (b["soc_max"] - b["soc"]) * E / b["eta_charge"] / dt))
                    b["soc"] += p * b["eta_charge"] * dt / E
                    charge += p
                elif cmd.action == "discharge":
                    floor = max(b["soc_min"], cmd.reserve_soc if cmd.reserve_soc is not None else b["soc_min"])
                    p = min(cmd.mw, b["power_mw"], max(0.0, (b["soc"] - floor) * E * b["eta_discharge"] / dt))
                    b["soc"] -= p * dt / (E * b["eta_discharge"])
                    discharge += p
            need = w["demand"] + pending[j] - dr - cut - sh - ren + charge - discharge
            if need > w["imp"]:  # emergency: dig into any battery energy down to soc_min
                for b in bats.values():
                    if need <= w["imp"] or b["status"] != "online":
                        continue
                    E = b["effective_energy_mwh"]
                    room = min(b["power_mw"], (b["soc"] - b["soc_min"]) * E * b["eta_discharge"] / dt)
                    p = max(0.0, min(room, need - w["imp"]))
                    b["soc"] -= p * dt / (E * b["eta_discharge"])
                    discharge += p
                    need -= p
            un = max(0.0, need - w["imp"])
            buy = min(max(need, 0.0), w["imp"])
            sell = min(max(-need, 0.0), w["exp"])
            deg = max((b["degradation_cost_per_mwh"] for b in bats.values()), default=0.0)
            cost += (buy * (w["price"] + w["grid_ef"] * w["carbon"]) - sell * w["price"] * self.sell_ratio
                     + dr * w["dr"] + cut * mk["noncritical_cut_cost"] + sh * mk["shift_fee"]
                     + un * mk["voll_noncritical"] + (charge + discharge) * deg) * dt
            unserved += un * dt
            dr_mwh += (dr + cut) * dt
            carbon_t += buy * w["grid_ef"] * dt
        # energy still owed (shifted past the horizon) and energy left in storage
        cost += float(pending[self.H:].sum()) * dt * self.end_value
        stored = sum(max(0.0, b["soc"] - b["soc_min"]) * b["effective_energy_mwh"] * b["eta_discharge"]
                     for b in bats.values())
        cost -= stored * self.end_value * 0.9
        return {"cost": cost, "unserved_mwh": unserved, "flex_mwh": dr_mwh, "co2_t": carbon_t}

    def evaluate(self, plans: list[Plan], risk_level: str = "low") -> list[dict]:
        w = STRESS_WEIGHT.get(risk_level, 0.2)
        rows = []
        for p in dict.fromkeys(plans):  # dedupe, keep order
            e, s = self.rollout(p, "expected"), self.rollout(p, "stress")
            rows.append({"plan_id": p.id, **asdict(p), "score": (1 - w) * e["cost"] + w * s["cost"],
                         "cost_expected": e["cost"], "cost_stress": s["cost"],
                         "unserved_expected_mwh": e["unserved_mwh"], "unserved_stress_mwh": s["unserved_mwh"],
                         "co2_expected_t": e["co2_t"]})
        rows.sort(key=lambda r: r["score"])
        return rows


def compact(rows: list[dict], n: int = 6) -> list[dict]:
    """Table for the LLM: costs in ₹ lakh, rounded, best first."""
    L = 1e5
    return [{"plan_id": r["plan_id"], "reserve_soc": r["reserve_soc"], "arbitrage": r["arbitrage"],
             "lookahead_h": r["lookahead_h"], "score_lakh": round(r["score"] / L, 2),
             "expected_lakh": round(r["cost_expected"] / L, 2), "stress_lakh": round(r["cost_stress"] / L, 2),
             "unserved_stress_mwh": round(r["unserved_stress_mwh"], 1)} for r in rows[:n]]
