"""The simulator: a virtual utility that advances in 15-minute steps.

Per step:
    1. observe()          -> what the agent can see: noisy nowcast, P10/P50/P90 forecasts, alerts
    2. agent decides      -> a Decision (battery, market, curtailment, demand, maintenance)
    3. apply_decision()   -> physics + protection + settlement:
         * clip impossible actions (and report every clip)
         * balance the bus: the grid tie absorbs any mismatch up to the line limits
         * beyond the limits, protection acts: cancel charging -> emergency battery (uses reserve)
           -> shed non-critical -> shed critical   (or cancel discharge -> absorb -> force curtail)
         * settle money: market trades at the exchange price, unscheduled deviations at
           penalty rates (doubled at low frequency), carbon, battery wear, DR, VoLL
    4. advance the clock, activate / resolve events, compute the next step's truth

Truth vs. what the agent sees: the agent never reads the "true" series directly. It gets a
nowcast with ~2% noise and forecasts whose error grows with the horizon. Surprise events are
invisible until they start. That is what makes planning under uncertainty (F2) meaningful.

Energy balance holds exactly every step (tests enforce it):
    renewables_used + battery_discharge + grid_import
        == demand_served + battery_charge + grid_export
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .assets import Battery, Portfolio, default_portfolio
from .clock import hhmm, label, start_of, to_datetime
from .constants import (
    AUTO_REPAIR_STEPS, DSM_DEFICIT_MIN_ADDER, DSM_DEFICIT_MULT, DSM_LOW_FREQ_MULT, DSM_SURPLUS_MULT,
    DT_H, EPS, FORECAST_HORIZON, HIGH_FREQ_HZ, INSPECTION_COST, INSPECTION_REPAIR_STEPS, LOW_FREQ_HZ,
    MAX_SHIFT_STEPS, NONCRITICAL_CUT_COST, PRICE_CAP, SELL_FACTOR, SHIFT_FEE, STEPS_PER_DAY,
    VOLL_CRITICAL, VOLL_NONCRITICAL,
)
from .decision import Decision, MaintenanceCommand
from .events import EVENT_CATALOG, Event, EventType, Window, describe, make_event
from .kpis import KpiAccumulator, components_from, weighted
from .profiles import generate_base_series
from .scenarios import Scenario

Z90 = 1.2816  # P10/P90 z-score

# forecast error model: sigma(lead) = s0 + s1 * sqrt(lead / 96)
FORECAST_SIGMA = {
    "solar": (0.04, 0.22),
    "wind": (0.06, 0.32),
    "demand": (0.01, 0.04),
    "price": (0.03, 0.15),
}
NOWCAST_SIGMA = {"solar": 0.02, "wind": 0.02, "demand": 0.01}


def _r(x, nd: int = 3):
    """Round floats / arrays for JSON."""
    if isinstance(x, np.ndarray):
        return np.round(x, nd).tolist()
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


class Simulator:
    def __init__(self, scenario: Scenario, seed: int | None = None, days: int = 1,
                 randomize: bool = False, keep_history: bool = True,
                 portfolio: Portfolio | None = None):
        self.scenario = scenario
        self.seed = int(scenario.seed if seed is None else seed)
        self.days = max(1, int(days))
        self.total_steps = self.days * STEPS_PER_DAY
        self.randomize = randomize
        self.keep_history = keep_history

        self.portfolio = portfolio.clone() if portfolio else default_portfolio()
        for b in self.portfolio.batteries:
            b.soc = float(scenario.initial_soc.get(b.id, 0.5))
            b.reserve_soc = b.soc_min
        self.names = {a.id: a.name for a in self.portfolio.all_assets()}

        weather_rng = np.random.default_rng([self.seed, 2002])
        self.params = scenario.weather.jittered(weather_rng) if randomize else scenario.weather
        # +1 day so a full 24 h forecast is always available
        self.base = generate_base_series(self.portfolio, self.params, self.days + 1, self.seed)
        self.start_time = start_of(scenario.start_date)

        self.events: list[Event] = []
        self._event_seq = 0
        event_rng = np.random.default_rng([self.seed, 3003])
        for spec in scenario.build_events(event_rng, self.days, randomize):
            if spec["start_step"] < self.total_steps:
                self.add_event(spec["type"], start_step=spec["start_step"],
                               duration_steps=spec.get("duration_steps"), params=spec.get("params"),
                               announce_step=spec.get("announce_step"), source="scenario")

        self.step = 0
        self.pending_shift: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.kpis = KpiAccumulator(battery_nameplate_mwh=sum(b.energy_mwh for b in self.portfolio.batteries))
        self.history: list[dict] = []
        self.last_tick: dict | None = None
        self.actual: dict | None = None
        self._obs_cache: dict | None = None
        self._z_cache: dict[int, dict[str, np.ndarray]] = {}
        self._begin_step()

    # ================================================================== events
    @property
    def done(self) -> bool:
        return self.step >= self.total_steps

    def add_event(self, etype: str, *, start_step: int, duration_steps: int | None,
                  params: dict | None = None, announce_step: int | None = None,
                  source: str = "scenario", severity: str | None = None) -> Event:
        try:
            ev = make_event(f"E{self._event_seq + 1:03d}", etype, start_step=start_step,
                            duration_steps=duration_steps, params=dict(params or {}),
                            announce_step=announce_step, source=source, severity=severity, names=self.names)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{etype}: invalid parameters ({e})") from e
        self._validate_event(ev)      # raises before anything is stored
        describe(ev, self.names)      # re-describe with the cleaned params
        self._event_seq += 1
        self.events.append(ev)
        return ev

    def _validate_event(self, ev: Event) -> None:
        """Type/range-check parameters and dry-run the effect, so a bad event can't break a run."""
        p, t, port = ev.params, ev.type, self.portfolio
        solar, wind = {s.id for s in port.solar}, {w.id for w in port.wind}
        bats, cons = {b.id for b in port.batteries}, {c.id for c in port.consumers}
        lines = {ln.id: ln for ln in port.lines}

        def num(key: str, lo: float | None, hi: float | None) -> None:
            if key not in p:
                return
            try:
                v = float(p[key])
            except (TypeError, ValueError):
                raise ValueError(f"{t}: '{key}' must be a number, got {p[key]!r}") from None
            if not np.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
                raise ValueError(f"{t}: '{key}'={v:g} must be within [{lo}, {hi}]")
            p[key] = v

        def one(key: str, allowed, optional: bool = False) -> None:
            if optional and p.get(key) is None:
                return
            if p.get(key) not in allowed:
                raise ValueError(f"{t}: unknown {key} '{p.get(key)}' (expected one of {sorted(allowed)})")

        def many(key: str, allowed) -> None:
            v = p.get(key)
            if not isinstance(v, list) or not v or any(x not in allowed for x in v):
                raise ValueError(f"{t}: '{key}' must be a non-empty list from {sorted(allowed)}")

        if t == EventType.CLOUD_COVER:
            many("sites", solar); num("drop", 0, 1)
        elif t == EventType.WIND_CHANGE:
            many("sites", wind); num("factor", 0, 5)
        elif t == EventType.PRICE_SPIKE:
            num("multiplier", 0.1, 10)
        elif t == EventType.ASSET_FAULT:
            one("asset_id", solar | wind | bats | set(lines))
        elif t == EventType.LINE_DERATE:
            one("line_id", lines)
            num("rating_mw", 0, lines[p["line_id"]].capacity_mw)
        elif t == EventType.DEMAND_SURGE:
            one("consumer_id", cons); num("delta_mw", -500, 500)
        elif t == EventType.FORECAST_UPDATE:
            one("target", {"solar", "wind"})
            many("sites", solar if p["target"] == "solar" else wind); num("factor", 0, 5)
        elif t == EventType.STORM:
            num("wind_factor", 0, 5); num("storm_wind_ms", 0, 60); num("solar_drop", 0, 1)
            num("frequency_offset_hz", -1, 1)
            one("line_id", lines, optional=True)
            if p.get("line_id"):
                num("line_rating_mw", 0, lines[p["line_id"]].capacity_mw)
        elif t == EventType.FREQUENCY_DIP:
            num("offset_hz", -1, 1)
        elif t == EventType.CARBON_PRICE_CHANGE:
            num("price", 0, 100_000)
        elif t == EventType.DR_INCENTIVE_CHANGE:
            num("rate", 0, 100_000)
        elif t == EventType.MAINTENANCE:
            one("asset_id", solar | wind | bats | set(lines))
            latest = self.total_steps - (ev.duration_steps or 1)
            p["flexible_until"] = int(np.clip(float(p.get("flexible_until", ev.start_step)),
                                              ev.start_step, max(ev.start_step, latest)))
        if ev.duration_steps is None and t not in (EventType.ASSET_FAULT, EventType.CARBON_PRICE_CHANGE,
                                                    EventType.DR_INCENTIVE_CHANGE):
            raise ValueError(f"{t}: duration_steps is required")
        if ev.start_step < 0:
            raise ValueError(f"{t}: start_step must be >= 0")
        a = min(ev.start_step, self.base.n - 1)
        probe = self._window(a, min(a + 2, self.base.n), known_at=None)
        try:
            ev.apply(probe)
            ev.apply(probe, known_at=a)
        except Exception as e:
            raise ValueError(f"{t}: parameters rejected ({type(e).__name__}: {e})") from None

    def inject_event(self, etype: str, params: dict | None = None, *, start_in_steps: int = 0,
                     duration_steps: int | None | str = "default", announce_in_steps: int | None = None,
                     source: str = "manual", severity: str | None = None) -> Event:
        """Add an event while the simulation runs (e.g. from a dashboard button)."""
        if etype not in EVENT_CATALOG:
            raise ValueError(f"Unknown event type '{etype}'")
        cat = EVENT_CATALOG[etype]
        dur = cat["duration_steps"] if duration_steps == "default" else duration_steps
        start = self.step + max(0, int(start_in_steps))
        if announce_in_steps is not None:
            announce = self.step + max(0, int(announce_in_steps))
        elif cat["announce_lead_steps"] is not None:
            announce = self.step  # forecast updates, storms, maintenance are announced right away
        else:
            announce = None       # surprise
        ev = self.add_event(etype, start_step=start, duration_steps=dur, params=params,
                            announce_step=announce, source=source, severity=severity)
        self._begin_step()        # recompute the current step so the effect is visible immediately
        return ev

    def _active_events(self, step: int, etype: str | None = None, asset_id: str | None = None):
        for ev in self.events:
            if ev.is_active(step) and (etype is None or ev.type == etype) and \
                    (asset_id is None or ev.params.get("asset_id") == asset_id):
                yield ev

    # ================================================================== world model
    def _window(self, a: int, b: int, known_at: int | None) -> Window:
        """Series for [a, b) with events applied. known_at=None -> truth; else what's known then."""
        bs = self.base
        b = min(b, bs.n)
        n = max(0, b - a)
        p = self.portfolio
        w = Window(
            a=a, b=b,
            solar_cf={k: v[a:b].copy() for k, v in bs.solar_cf.items()},
            wind_speed={k: v[a:b].copy() for k, v in bs.wind_speed.items()},
            demand={k: v[a:b].copy() for k, v in bs.demand.items()},
            price=bs.price[a:b].copy(), grid_ef=bs.grid_ef[a:b].copy(),
            frequency=bs.frequency[a:b].copy(), carbon_price=bs.carbon_price[a:b].copy(),
            dr_incentive=bs.dr_incentive[a:b].copy(), temperature_c=bs.temperature_c[a:b].copy(),
            cloudiness=bs.cloudiness[a:b].copy(),
            available={x.id: np.ones(n) for x in (*p.solar, *p.wind, *p.batteries)},
            line_rating={ln.id: np.full(n, float(ln.capacity_mw)) for ln in p.lines},
        )
        for ev in self.events:
            if known_at is None or ev.is_known(known_at):
                ev.apply(w, known_at)
        return w

    def _materialize(self, w: Window) -> dict:
        p = self.portfolio
        # potential = what the farm would produce if it were available (used to plan maintenance)
        solar_pot = {s.id: s.capacity_mw * w.solar_cf[s.id] for s in p.solar}
        wind_pot = {f.id: f.capacity_mw * f.capacity_factor(w.wind_speed[f.id]) for f in p.wind}
        solar = {k: v * w.available[k] for k, v in solar_pot.items()}
        wind = {k: v * w.available[k] for k, v in wind_pot.items()}
        rating = w.line_rating
        limit = sum(rating.values()) if rating else np.zeros(w.b - w.a)
        return {
            "solar_mw": solar, "wind_mw": wind, "solar_potential_mw": solar_pot, "wind_potential_mw": wind_pot,
            "demand_mw": w.demand,
            "wind_speed": w.wind_speed,
            "battery_available": {b.id: w.available[b.id] for b in p.batteries},
            "line_rating": rating, "tie_limit": limit,
            "price": w.price, "grid_ef": w.grid_ef, "frequency": w.frequency,
            "carbon_price": w.carbon_price, "dr_incentive": w.dr_incentive,
            "temperature_c": w.temperature_c, "cloudiness": w.cloudiness,
        }

    def _begin_step(self) -> None:
        """Activate/resolve events and compute the truth for the current step (idempotent)."""
        self._obs_cache = None
        if self.done:
            self.actual = None
            return
        t = self.step
        for ev in self.events:  # faults clear after a crew visit, or eventually on their own
            if ev.type == EventType.ASSET_FAULT and ev.resolved_step is None and t > ev.start_step:
                eta = ev.params.get("repair_eta_step")
                if (eta is not None and t >= eta) or t >= ev.start_step + AUTO_REPAIR_STEPS:
                    ev.resolved_step = t
        m = self._materialize(self._window(t, t + 1, known_at=None))
        scalar = lambda d: {k: float(v[0]) for k, v in d.items()}  # noqa: E731
        self.actual = {
            "solar_mw": scalar(m["solar_mw"]), "wind_mw": scalar(m["wind_mw"]),
            "demand_mw": scalar(m["demand_mw"]), "wind_speed": scalar(m["wind_speed"]),
            "battery_available": scalar(m["battery_available"]),
            "line_rating": scalar(m["line_rating"]),
            **{k: float(m[k][0]) for k in ("price", "grid_ef", "frequency", "carbon_price",
                                            "dr_incentive", "temperature_c", "cloudiness")},
        }
        # asset statuses
        for a in (*self.portfolio.solar, *self.portfolio.wind, *self.portfolio.batteries):
            if any(True for _ in self._active_events(t, EventType.ASSET_FAULT, a.id)):
                a.status = "faulted"
            elif any(True for _ in self._active_events(t, EventType.MAINTENANCE, a.id)):
                a.status = "maintenance"
            else:
                a.status = "online"
        for ln in self.portfolio.lines:
            ln.rating_mw = self.actual["line_rating"][ln.id]
            ln.status = ("tripped" if ln.rating_mw <= EPS else
                         "derated" if ln.rating_mw < ln.capacity_mw - EPS else "online")

    # ================================================================== observation
    def _z(self, issue: int) -> dict[str, np.ndarray]:
        """Forecast error paths, fixed per hourly forecast issue (so forecasts don't flicker)."""
        if issue not in self._z_cache:
            out = {}
            for i, name in enumerate(FORECAST_SIGMA):
                rng = np.random.default_rng([self.seed, issue, 500 + i])
                eps = rng.normal(0, 1, FORECAST_HORIZON + 8)
                z = np.empty_like(eps)
                z[0] = eps[0]
                for k in range(1, len(eps)):
                    z[k] = 0.9 * z[k - 1] + np.sqrt(1 - 0.81) * eps[k]
                out[name] = z
            if len(self._z_cache) > 64:
                self._z_cache.clear()
            self._z_cache[issue] = out
        return self._z_cache[issue]

    def _forecast(self, t: int) -> dict:
        a = t + 1
        b = min(a + FORECAST_HORIZON, self.base.n)
        H = b - a
        m = self._materialize(self._window(a, b, known_at=t))
        issue = t - t % 4
        z = self._z(issue)
        lead_issue = np.arange(a, b) - issue
        lead_now = np.arange(1, H + 1)

        def sig(name, lead):
            s0, s1 = FORECAST_SIGMA[name]
            return s0 + s1 * np.sqrt(lead / FORECAST_HORIZON)

        def band(name, true, cap=None):
            p50 = true * (1 + sig(name, lead_issue) * z[name][lead_issue])
            hi = np.inf if cap is None else cap
            p50 = np.clip(p50, 0, hi)
            s = sig(name, lead_now)
            return {"p10": _r(np.clip(p50 * (1 - Z90 * s), 0, hi)), "p50": _r(p50),
                    "p90": _r(np.clip(p50 * (1 + Z90 * s), 0, hi))}

        solar_true = sum(m["solar_mw"].values())
        wind_true = sum(m["wind_mw"].values())
        shifted = np.array([sum(self.pending_shift.get(s, {}).values()) for s in range(a, b)]) \
            if self.pending_shift else np.zeros(H)
        demand_true = sum(m["demand_mw"].values())
        demand_band = band("demand", demand_true)
        if shifted.any():  # our own scheduled (shifted) load is known exactly
            for k in demand_band:
                demand_band[k] = _r(np.array(demand_band[k]) + shifted)
        solar_factor = 1 + sig("solar", lead_issue) * z["solar"][lead_issue]
        wind_factor = 1 + sig("wind", lead_issue) * z["wind"][lead_issue]
        return {
            "start_step": a,
            "horizon_steps": H,
            "times": [hhmm(s) for s in range(a, b)],
            "solar_total_mw": band("solar", solar_true, self.portfolio.solar_capacity_mw),
            "wind_total_mw": band("wind", wind_true, self.portfolio.wind_capacity_mw),
            "demand_total_mw": demand_band,
            "price": band("price", m["price"], PRICE_CAP),
            "solar_by_farm_p50": {k: _r(np.clip(v * solar_factor, 0, None), 2) for k, v in m["solar_mw"].items()},
            "wind_by_farm_p50": {k: _r(np.clip(v * wind_factor, 0, None), 2) for k, v in m["wind_mw"].items()},
            # same, but ignoring planned outages: what a maintenance window would cost
            "solar_potential_by_farm_p50": {k: _r(np.clip(v * solar_factor, 0, None), 2)
                                            for k, v in m["solar_potential_mw"].items()},
            "wind_potential_by_farm_p50": {k: _r(np.clip(v * wind_factor, 0, None), 2)
                                           for k, v in m["wind_potential_mw"].items()},
            "grid_ef": _r(m["grid_ef"]),
            "carbon_price": _r(m["carbon_price"], 0),
            "dr_incentive": _r(m["dr_incentive"], 0),
            "import_limit_mw": _r(m["tie_limit"], 1),
            "export_limit_mw": _r(m["tie_limit"], 1),
            "battery_available": {k: _r(v, 0) for k, v in m["battery_available"].items()},
            "shifted_load_mw": _r(shifted, 2),
        }

    def _nowcast(self, t: int) -> dict:
        rng = np.random.default_rng([self.seed, t, 77])
        act = self.actual
        noisy = lambda v, s: max(0.0, v * (1 + rng.normal(0, s)))  # noqa: E731
        return {
            "solar": {k: noisy(v, NOWCAST_SIGMA["solar"]) for k, v in act["solar_mw"].items()},
            "wind": {k: noisy(v, NOWCAST_SIGMA["wind"]) for k, v in act["wind_mw"].items()},
            "demand": {k: noisy(v, NOWCAST_SIGMA["demand"]) for k, v in act["demand_mw"].items()},
        }

    def alerts(self, t: int | None = None) -> list[dict]:
        t = self.step if t is None else t
        out = []
        for ev in self.events:
            if not ev.is_known(t):
                continue
            st = ev.status(t)
            if st == "ended" or st == "cancelled":
                continue
            if ev.duration_steps is None and ev.type in (EventType.CARBON_PRICE_CHANGE,
                                                         EventType.DR_INCENTIVE_CHANGE) \
                    and t > ev.start_step + 8:
                continue  # persistent info events only shown for 2 h after they start
            d = ev.to_dict(t)
            d["starts_in_steps"] = max(0, ev.start_step - t)
            out.append(d)
        return out

    def observe(self) -> dict:
        """Everything the agent is allowed to know at the current step."""
        if self._obs_cache is not None:
            return self._obs_cache
        if self.done:
            raise RuntimeError("Simulation finished")
        t, act, p = self.step, self.actual, self.portfolio
        now = self._nowcast(t)
        firm_in = self.pending_shift.get(t, {})

        solar = {s.id: {"name": s.name, "available_mw": _r(now["solar"][s.id]), "capacity_mw": s.capacity_mw,
                        "status": s.status, "region": s.region} for s in p.solar}
        wind = {f.id: {"name": f.name, "available_mw": _r(now["wind"][f.id]), "capacity_mw": f.capacity_mw,
                       "wind_speed_ms": _r(act["wind_speed"][f.id], 2), "cut_out_ms": f.cut_out_ms,
                       "status": f.status, "region": f.region} for f in p.wind}
        demand = {}
        for c in p.consumers:
            base = now["demand"][c.id]
            dr_ok = c.dr_steps_used_today < c.dr_max_steps_per_day
            demand[c.id] = {
                "name": c.name, "kind": c.kind,
                "demand_mw": _r(base + firm_in.get(c.id, 0.0)),
                "firm_shifted_in_mw": _r(firm_in.get(c.id, 0.0)),
                "critical_mw": _r(c.critical_fraction * base),
                "dr_available_mw": _r(c.dr_fraction * base if dr_ok else 0.0),
                "shiftable_mw": _r(c.shiftable_fraction * base if t < self.total_steps - 1 else 0.0),
                "noncritical_mw": _r((1 - c.critical_fraction) * base),
                "dr_steps_left": c.dr_max_steps_per_day - c.dr_steps_used_today,
                "tariff": c.tariff_per_mwh,
            }
        batteries = {}
        for b in p.batteries:
            batteries[b.id] = {
                "name": b.name, "status": b.status, "soc": _r(b.soc, 4), "stored_mwh": _r(b.stored_mwh),
                "energy_mwh": b.energy_mwh, "effective_energy_mwh": _r(b.effective_energy_mwh),
                "power_mw": b.power_mw, "soc_min": b.soc_min, "soc_max": b.soc_max,
                "reserve_soc": _r(b.reserve_soc, 4),
                "max_charge_mw": _r(b.max_charge_mw()),
                "max_discharge_mw": _r(b.max_discharge_mw(floor_soc=b.reserve_soc)),
                "max_discharge_emergency_mw": _r(b.max_discharge_mw()),
                "eta_charge": b.eta_charge, "eta_discharge": b.eta_discharge, "soh": _r(b.soh, 5),
                "degradation_cost_per_mwh": b.degradation_cost_per_mwh,
                "clean_fraction": _r(b.clean_fraction),
            }
        limit = sum(act["line_rating"].values())
        price = act["price"]
        low = act["frequency"] < LOW_FREQ_HZ
        solar_total = sum(now["solar"].values())
        wind_total = sum(now["wind"].values())
        demand_total = sum(v["demand_mw"] for v in demand.values())
        obs = {
            "step": t, "time": hhmm(t), "label": label(t), "day": t // STEPS_PER_DAY,
            "timestamp": to_datetime(self.start_time, t).isoformat(),
            "steps_remaining": self.total_steps - t, "dt_hours": DT_H,
            "current": {
                "solar": solar, "wind": wind, "demand": demand, "batteries": batteries,
                "solar_total_mw": _r(solar_total), "wind_total_mw": _r(wind_total),
                "renewable_total_mw": _r(solar_total + wind_total),
                "demand_total_mw": _r(demand_total),
                "net_position_mw": _r(solar_total + wind_total - demand_total),
                "grid": {
                    "frequency_hz": _r(act["frequency"], 3),
                    "lines": {ln.id: {"name": ln.name, "capacity_mw": ln.capacity_mw,
                                      "rating_mw": _r(ln.rating_mw, 1), "status": ln.status}
                              for ln in p.lines},
                    "import_limit_mw": _r(limit, 1), "export_limit_mw": _r(limit, 1),
                },
                "market": {
                    "price": _r(price, 1), "sell_price": _r(price * SELL_FACTOR, 1),
                    "dsm_deficit_rate": _r(max(price * DSM_DEFICIT_MULT, price + DSM_DEFICIT_MIN_ADDER)
                                           * (DSM_LOW_FREQ_MULT if low else 1), 1),
                    "dsm_surplus_rate": _r(0.0 if act["frequency"] > HIGH_FREQ_HZ else price * DSM_SURPLUS_MULT, 1),
                    "carbon_price": _r(act["carbon_price"], 1), "grid_ef": _r(act["grid_ef"], 4),
                    "dr_incentive": _r(act["dr_incentive"], 1), "price_cap": PRICE_CAP,
                    "shift_fee": SHIFT_FEE, "noncritical_cut_cost": NONCRITICAL_CUT_COST,
                    "voll_noncritical": VOLL_NONCRITICAL, "voll_critical": VOLL_CRITICAL,
                },
                "weather": {"temperature_c": _r(act["temperature_c"], 1),
                            "cloudiness": _r(act["cloudiness"], 2),
                            "wind_speed_avg_ms": _r(float(np.mean(list(act["wind_speed"].values()))), 2)},
            },
            "forecast": self._forecast(t),
            "alerts": self.alerts(t),
            "maintenance": [ev.to_dict(t) for ev in self.events
                            if ev.type == EventType.MAINTENANCE and ev.status(t) in ("planned", "active")],
            "pending_shifted_load": [
                {"step": s, "time": hhmm(s), "consumer_id": cid, "mw": _r(mw)}
                for s, d in sorted(self.pending_shift.items()) if s >= t for cid, mw in d.items() if mw > EPS
            ],
        }
        self._obs_cache = obs
        return obs

    # ================================================================== actions
    def _apply_maintenance(self, cmds: list[MaintenanceCommand], note, kd) -> list[dict]:
        t = self.step
        results = []
        for m in cmds:
            if m.action == "dispatch_inspection":
                fault = next((ev for ev in self._active_events(t, EventType.ASSET_FAULT, m.asset_id)), None)
                if fault is None:
                    note("clipped", f"No active fault on {m.asset_id}; no crew dispatched", m.asset_id)
                    continue
                if fault.params.get("repair_eta_step") is not None:
                    continue  # already on the way
                eta = t + INSPECTION_REPAIR_STEPS
                natural_end = fault.end_step() if fault.duration_steps is not None \
                    else fault.start_step + AUTO_REPAIR_STEPS
                if eta >= natural_end:
                    note("clipped", f"{m.asset_id} will be back by {hhmm(natural_end)} anyway; crew not sent",
                         m.asset_id)
                    continue
                if fault.duration_steps is not None:  # scripted fault with a fixed length: crew shortens it
                    fault.duration_steps = eta - fault.start_step
                fault.params["repair_eta_step"] = eta
                fault.params["crew_dispatched_step"] = t
                kd["maintenance_cost"] += INSPECTION_COST
                note("action", f"Inspection crew dispatched to {m.asset_id}; back online ~{hhmm(eta)}",
                     m.asset_id)
                results.append({"action": "dispatch_inspection", "asset_id": m.asset_id, "eta_step": eta})
            elif m.action in ("delay", "schedule"):
                ev = None
                for cand in self.events:
                    if cand.type != EventType.MAINTENANCE:
                        continue
                    if (m.maintenance_id and cand.id == m.maintenance_id) or \
                            (not m.maintenance_id and cand.params.get("asset_id") == m.asset_id
                             and cand.start_step > t):
                        ev = cand
                        break
                if ev is None or ev.start_step <= t:
                    note("clipped", f"Maintenance {m.maintenance_id or m.asset_id} not found or already started")
                    continue
                if m.action == "delay":
                    new_start = ev.start_step + int(m.delay_steps or 0)
                else:
                    new_start = int(m.start_step if m.start_step is not None else ev.start_step)
                latest = int(ev.params.get("flexible_until", ev.start_step))
                clamped = int(np.clip(new_start, t + 1, max(latest, t + 1)))
                if clamped != new_start:
                    note("clipped", f"Maintenance {ev.id} start limited to {hhmm(clamped)} (flexibility window)",
                         ev.params.get("asset_id"))
                if clamped != ev.start_step:
                    old = ev.start_step
                    ev.start_step = clamped
                    describe(ev, self.names)
                    note("action", f"Maintenance {ev.id} on {ev.params.get('asset_id')} moved "
                                   f"{hhmm(old)} -> {hhmm(clamped)}", ev.params.get("asset_id"))
                    results.append({"action": m.action, "maintenance_id": ev.id,
                                    "from_step": old, "to_step": clamped})
            else:
                note("clipped", f"Unknown maintenance action '{m.action}'")
        return results

    def _expand_curtailment(self, req: dict[str, float], avail: dict[str, float], note) -> dict[str, float]:
        out = {aid: 0.0 for aid in avail}
        for key, mw in req.items():
            if mw <= 0:
                continue
            if key in ("solar", "wind"):
                ids = [a for a in avail if self.portfolio.kind_of(a) == key]
                tot = sum(avail[a] for a in ids)
                for a in ids:
                    out[a] += mw * (avail[a] / tot) if tot > EPS else 0.0
            elif key in avail:
                out[key] += mw
            else:
                note("clipped", f"Curtailment target '{key}' unknown")
        for aid in out:
            if out[aid] > avail[aid] + 1e-3:
                note("clipped", f"Curtail {aid} {out[aid]:.1f} MW > available {avail[aid]:.1f} MW", aid)
            out[aid] = min(out[aid], avail[aid])
        return out

    def apply_decision(self, decision: Decision | dict | None = None, *, latency_ms: float = 0.0,
                       fallback: bool = False) -> dict:
        if self.done:
            raise RuntimeError("Simulation finished")
        if decision is None:
            decision = Decision.idle()
        # normalise and type-check everything up front, so nothing is mutated if the input is bad
        try:
            decision = Decision.from_dict(decision.to_dict() if isinstance(decision, Decision) else decision)
        except Exception as e:
            raise ValueError(f"Invalid decision: {type(e).__name__}: {e}") from e
        t, act, p, dt = self.step, self.actual, self.portfolio, DT_H
        notes: list[dict] = []
        kd: dict[str, float] = defaultdict(float)

        def note(kind: str, message: str, asset: str | None = None):
            notes.append({"kind": kind, "message": message, "asset": asset})
            if kind == "clipped":
                kd["clipped_actions"] += 1
            elif kind == "emergency":
                kd["emergency_actions"] += 1

        # ---- 1. maintenance and crews (affect future steps)
        maint = self._apply_maintenance(decision.maintenance, note, kd)

        # ---- 2. reserve floors
        known_batteries = {b.id for b in p.batteries}
        for bid, cmd in decision.battery.items():
            if bid not in known_batteries:
                note("clipped", f"Unknown battery '{bid}'")
                continue
            if cmd.reserve_soc is not None:
                b = p.get(bid)
                b.reserve_soc = float(np.clip(cmd.reserve_soc, b.soc_min, b.soc_max))

        # ---- 3. generation and voluntary curtailment
        avail = {**act["solar_mw"], **act["wind_mw"]}
        curtail = self._expand_curtailment(decision.curtail_mw, avail, note)
        used = {aid: avail[aid] - curtail[aid] for aid in avail}
        forced = {aid: 0.0 for aid in avail}

        # ---- 4. demand side
        firm_in = self.pending_shift.get(t, {})
        cons = {}
        for key in (*decision.demand_response_mw, *decision.reduce_noncritical_mw,
                    *(s.consumer_id for s in decision.shift_load)):
            if p.kind_of(key) != "consumer":
                note("clipped", f"Unknown consumer '{key}'")
        for c in p.consumers:
            base = act["demand_mw"][c.id]
            fin = float(firm_in.get(c.id, 0.0))
            dr_req = decision.demand_response_mw.get(c.id, 0.0)
            dr_cap = c.dr_fraction * base if c.dr_steps_used_today < c.dr_max_steps_per_day else 0.0
            dr = min(dr_req, dr_cap)
            if dr_req > dr_cap + 1e-3:
                note("clipped", f"DR at {c.id} limited to {dr_cap:.1f} MW", c.id)
            shift_cmds = []
            for s in decision.shift_load:
                if s.consumer_id != c.id or s.mw <= 0:
                    continue
                delay = int(np.clip(s.delay_steps, 1, MAX_SHIFT_STEPS))
                if t + delay > self.total_steps - 1:
                    note("clipped", f"Shift at {c.id} to {hhmm(t + delay)} is past the end of the run; ignored", c.id)
                    continue
                shift_cmds.append((s, delay))
            shift_req = sum(s.mw for s, _ in shift_cmds)
            shift_cap = c.shiftable_fraction * base
            shift = min(shift_req, shift_cap)
            if shift_req > shift_cap + 1e-3:
                note("clipped", f"Load shift at {c.id} limited to {shift_cap:.1f} MW", c.id)
            nc_req = decision.reduce_noncritical_mw.get(c.id, 0.0)
            nc_cap = max(0.0, (1 - c.critical_fraction) * base - dr - shift)
            nc = min(nc_req, nc_cap)
            if nc_req > nc_cap + 1e-3:
                note("clipped", f"Non-critical cut at {c.id} limited to {nc_cap:.1f} MW", c.id)
            if dr > EPS:
                c.dr_steps_used_today += 1
            if shift > EPS:
                scale = shift / shift_req
                for s, delay in shift_cmds:
                    self.pending_shift[t + delay][c.id] += s.mw * scale
            after = base - dr - shift - nc + fin
            cons[c.id] = {"base": base, "firm_in": fin, "dr": dr, "shift": shift, "nc": nc,
                          "after": after, "critical": min(after, c.critical_fraction * base),
                          "shed_nc": 0.0, "shed_c": 0.0}

        # ---- 5. battery commands (agent level: cannot dip below the reserve floor)
        ch = {b.id: 0.0 for b in p.batteries}
        dis = {b.id: 0.0 for b in p.batteries}
        for b in p.batteries:
            cmd = decision.battery.get(b.id)
            if cmd is None or cmd.action == "idle" or cmd.mw <= 0:
                continue
            if not b.online:
                note("clipped", f"{b.id} is {b.status}; command ignored", b.id)
                continue
            if cmd.action == "charge":
                lim = b.max_charge_mw()
                ch[b.id] = min(cmd.mw, lim)
            else:
                lim = b.max_discharge_mw(floor_soc=b.reserve_soc)
                dis[b.id] = min(cmd.mw, lim)
            if cmd.mw > lim + 1e-3:
                note("clipped", f"{b.id} {cmd.action} {cmd.mw:.1f} MW limited to {lim:.1f} MW", b.id)

        # ---- 6. market schedule
        limit = float(sum(act["line_rating"].values()))
        buy, sell = decision.buy_mw, decision.sell_mw
        if buy > EPS and sell > EPS:
            note("clipped", f"Buy {buy:.1f} and sell {sell:.1f} MW netted")
        sched = buy - sell
        sched_clipped = float(np.clip(sched, -limit, limit))
        if abs(sched_clipped - sched) > 1e-3:
            note("clipped", f"Trade {sched:+.1f} MW limited to tie-line capacity {limit:.1f} MW")
        sched = sched_clipped

        # ---- 7. physical balance and protection
        def required_import() -> float:
            served = sum(c["after"] - c["shed_nc"] - c["shed_c"] for c in cons.values())
            return served + sum(ch.values()) - sum(used.values()) - sum(dis.values())

        emergency = {b.id: 0.0 for b in p.batteries}
        F = required_import()
        if F > limit + EPS:
            short = F - limit
            for aid in used:  # 0) give back renewables the agent chose to curtail
                r = min(curtail[aid], short)
                if r > EPS:
                    curtail[aid] -= r
                    used[aid] += r
                    short -= r
                    note("emergency", f"Grid tie full: curtailment of {aid} undone ({r:.1f} MW)", aid)
            for b in p.batteries:  # a) stop charging
                r = min(ch[b.id], short)
                if r > EPS:
                    ch[b.id] -= r
                    short -= r
                    note("emergency", f"Grid tie full: {b.id} charging cut by {r:.1f} MW", b.id)
            for b in sorted(p.batteries, key=lambda x: -x.stored_mwh):  # b) emergency discharge
                if short <= EPS:
                    break
                r = min(b.max_discharge_mw(floor_soc=b.soc_min, already_mw=dis[b.id]), short)
                if r > EPS:
                    dis[b.id] += r
                    emergency[b.id] += r
                    short -= r
                    note("emergency", f"{b.id} emergency discharge {r:.1f} MW (reserve used)", b.id)
            for level in ("nc", "c"):  # c) shed load, non-critical first
                if short <= EPS:
                    break
                room = {cid: (c["after"] - c["critical"] - c["shed_nc"]) if level == "nc"
                        else (c["critical"] - c["shed_c"]) for cid, c in cons.items()}
                total = sum(max(0.0, v) for v in room.values())
                if total <= EPS:
                    continue
                take = min(short, total)
                for cid, v in room.items():
                    if v > EPS:
                        cons[cid]["shed_" + level] += take * v / total
                short -= take
                note("emergency", f"LOAD SHED {take:.1f} MW ({'non-critical' if level == 'nc' else 'CRITICAL'})")
        elif F < -limit - EPS:
            excess = -limit - F
            for b in p.batteries:  # a) stop discharging
                r = min(dis[b.id], excess)
                if r > EPS:
                    dis[b.id] -= r
                    excess -= r
                    note("emergency", f"Export limit: {b.id} discharge cut by {r:.1f} MW", b.id)
            for b in sorted(p.batteries, key=lambda x: x.stored_mwh):  # b) absorb into storage
                if excess <= EPS:
                    break
                r = min(b.max_charge_mw(already_mw=ch[b.id]), excess)
                if r > EPS:
                    ch[b.id] += r
                    emergency[b.id] += r
                    excess -= r
                    note("emergency", f"{b.id} absorbing {r:.1f} MW surplus", b.id)
            if excess > EPS:  # c) forced curtailment, pro rata
                total = sum(used.values())
                for aid in used:
                    r = excess * used[aid] / total if total > EPS else 0.0
                    used[aid] -= r
                    forced[aid] += r
                note("emergency", f"Forced curtailment {excess:.1f} MW (export limit {limit:.0f} MW)")
        F = required_import()
        if abs(F) > limit + 1e-6:  # numerical guard
            F = float(np.clip(F, -limit, limit))

        # ---- 8. settlement
        price, freq = act["price"], act["frequency"]
        low, high = freq < LOW_FREQ_HZ, freq > HIGH_FREQ_HZ
        dsm_def = max(price * DSM_DEFICIT_MULT, price + DSM_DEFICIT_MIN_ADDER) * (DSM_LOW_FREQ_MULT if low else 1)
        dsm_sur = 0.0 if high else price * DSM_SURPLUS_MULT
        sell_price = price * SELL_FACTOR
        deviation = F - sched
        imports, exports = max(F, 0.0), max(-F, 0.0)

        # ---- 9. batteries and clean-energy provenance
        gen_used = sum(used.values())
        clean_src = gen_used + sum(dis[b.id] * b.clean_fraction for b in p.batteries)
        total_src = gen_used + sum(dis.values()) + imports
        mix = clean_src / total_src if total_src > EPS else 1.0
        bat_out = {}
        deg_cost = 0.0
        for b in p.batteries:
            soc0 = b.soc
            res = b.apply(ch[b.id], dis[b.id], mix)
            deg_cost += res["degradation_cost"]
            bat_out[b.id] = {"charge_mw": _r(ch[b.id]), "discharge_mw": _r(dis[b.id]),
                             "emergency_mw": _r(emergency[b.id]), "soc_before": _r(soc0, 4),
                             "soc": _r(b.soc, 4), "reserve_soc": _r(b.reserve_soc, 4),
                             "status": b.status, "soh": _r(b.soh, 6)}

        # ---- 10. lines
        ratings = act["line_rating"]
        loading = (abs(F) / limit * 100) if limit > EPS else (0.0 if abs(F) < EPS else 100.0)
        lines = {lid: {"flow_mw": _r(F * r / limit if limit > EPS else 0.0, 2), "rating_mw": _r(r, 1),
                       "loading_pct": _r(loading, 1)} for lid, r in ratings.items()}

        # ---- 11. KPI deltas (₹, MWh, t)
        shed_nc = sum(c["shed_nc"] for c in cons.values())
        shed_c = sum(c["shed_c"] for c in cons.values())
        served_mw = sum(c["after"] - c["shed_nc"] - c["shed_c"] for c in cons.values())
        requested_mw = sum(c["base"] + c["firm_in"] for c in cons.values())
        tariffs = {c.id: c.tariff_per_mwh for c in p.consumers}
        kd.update({
            # original demand, counted once (load shifted in from earlier is not new demand)
            "demand_requested_mwh": sum(c["base"] for c in cons.values()) * dt,
            "demand_served_mwh": served_mw * dt,
            "unserved_noncritical_mwh": shed_nc * dt,
            "unserved_critical_mwh": shed_c * dt,
            "dr_mwh": sum(c["dr"] for c in cons.values()) * dt,
            "shifted_mwh": sum(c["shift"] for c in cons.values()) * dt,
            "noncritical_cut_mwh": sum(c["nc"] for c in cons.values()) * dt,
            "renewable_available_mwh": sum(avail.values()) * dt,
            "renewable_used_mwh": gen_used * dt,
            "curtailed_voluntary_mwh": sum(curtail.values()) * dt,
            "curtailed_forced_mwh": sum(forced.values()) * dt,
            "import_mwh": imports * dt,
            "export_mwh": exports * dt,
            "scheduled_buy_mwh": max(sched, 0.0) * dt,
            "scheduled_sell_mwh": max(-sched, 0.0) * dt,
            "deviation_import_mwh": max(deviation, 0.0) * dt,
            "deviation_export_mwh": max(-deviation, 0.0) * dt,
            "battery_charge_mwh": sum(ch.values()) * dt,
            "battery_discharge_mwh": sum(dis.values()) * dt,
            "emergency_battery_mwh": sum(emergency.values()) * dt,
            "clean_to_demand_mwh": served_mw * mix * dt,
            "low_freq_import_mwh": imports * dt if low else 0.0,
            "co2_t": imports * dt * act["grid_ef"],
            "avoided_co2_t": exports * dt * act["grid_ef"],
            "purchase_cost": max(sched, 0.0) * dt * price,
            "sales_revenue": max(-sched, 0.0) * dt * sell_price,
            "dsm_cost": max(deviation, 0.0) * dt * dsm_def,
            "dsm_credit": max(-deviation, 0.0) * dt * dsm_sur,
            "carbon_cost": imports * dt * act["grid_ef"] * act["carbon_price"],
            "degradation_cost": deg_cost,
            "dr_cost": sum(c["dr"] for c in cons.values()) * dt * act["dr_incentive"],
            "shift_cost": sum(c["shift"] for c in cons.values()) * dt * SHIFT_FEE,
            "noncritical_cost": sum(c["nc"] for c in cons.values()) * dt * NONCRITICAL_CUT_COST,
            "shed_penalty": (shed_nc * VOLL_NONCRITICAL + shed_c * VOLL_CRITICAL) * dt,
            "tariff_revenue": sum((c["after"] - c["shed_nc"] - c["shed_c"]) * tariffs[cid]
                                  for cid, c in cons.items()) * dt,
            "lost_tariff_flex": sum((c["dr"] + c["nc"]) * tariffs[cid] for cid, c in cons.items()) * dt,
            "lost_tariff_shed": sum((c["shed_nc"] + c["shed_c"]) * tariffs[cid] for cid, c in cons.items()) * dt,
            "line_loading_pct": loading,
            "steps_at_line_limit": 1.0 if loading >= 99.0 else 0.0,
            "shed_steps": 1.0 if (shed_nc + shed_c) > EPS else 0.0,
            "fallbacks": 1.0 if fallback else 0.0,
            "decision_latency_ms_total": float(latency_ms),
        })
        comps = components_from(kd)
        default_total = weighted(comps, None)
        agent_total = weighted(comps, decision.objective_weights)
        kd["objective_weighted_total"] = agent_total
        money_keys = ("purchase_cost", "sales_revenue", "dsm_cost", "dsm_credit", "carbon_cost",
                      "degradation_cost", "dr_cost", "shift_cost", "noncritical_cost", "shed_penalty",
                      "maintenance_cost", "tariff_revenue")

        solar_used = sum(used[s.id] for s in p.solar)
        wind_used = sum(used[w.id] for w in p.wind)
        tick = {
            "step": t, "time": hhmm(t), "label": label(t),
            "timestamp": to_datetime(self.start_time, t).isoformat(),
            "agent": decision.agent, "mode": decision.mode, "reasons": decision.reasons,
            "fallback": fallback, "latency_ms": _r(latency_ms, 1),
            "decision": decision.to_dict(),
            "applied": {
                "batteries": bat_out,
                "market": {"buy_mw": _r(max(sched, 0.0)), "sell_mw": _r(max(-sched, 0.0)),
                           "scheduled_net_mw": _r(sched)},
                "curtail_mw": {k: _r(v) for k, v in curtail.items() if v > EPS},
                "forced_curtail_mw": {k: _r(v) for k, v in forced.items() if v > EPS},
                "demand_response_mw": {k: _r(c["dr"]) for k, c in cons.items() if c["dr"] > EPS},
                "shift_mw": {k: _r(c["shift"]) for k, c in cons.items() if c["shift"] > EPS},
                "noncritical_cut_mw": {k: _r(c["nc"]) for k, c in cons.items() if c["nc"] > EPS},
                "shed_mw": {k: {"noncritical": _r(c["shed_nc"]), "critical": _r(c["shed_c"])}
                            for k, c in cons.items() if c["shed_nc"] + c["shed_c"] > EPS},
                "maintenance": maint,
            },
            "flows": {
                "solar_available_mw": _r(sum(act["solar_mw"].values())),
                "wind_available_mw": _r(sum(act["wind_mw"].values())),
                "solar_used_mw": _r(solar_used), "wind_used_mw": _r(wind_used),
                "renewable_used_mw": _r(gen_used),
                "curtailed_mw": _r(sum(curtail.values())), "forced_curtailed_mw": _r(sum(forced.values())),
                "demand_requested_mw": _r(requested_mw), "demand_served_mw": _r(served_mw),
                "unserved_mw": _r(shed_nc + shed_c),
                "dr_mw": _r(sum(c["dr"] for c in cons.values())),
                "shifted_out_mw": _r(sum(c["shift"] for c in cons.values())),
                "shifted_in_mw": _r(sum(c["firm_in"] for c in cons.values())),
                "noncritical_cut_mw": _r(sum(c["nc"] for c in cons.values())),
                "battery_charge_mw": _r(sum(ch.values())), "battery_discharge_mw": _r(sum(dis.values())),
                "battery_net_mw": _r(sum(dis.values()) - sum(ch.values())),
                "grid_net_import_mw": _r(F), "scheduled_net_mw": _r(sched), "deviation_mw": _r(deviation),
                "import_limit_mw": _r(limit, 1), "export_limit_mw": _r(limit, 1),
                "line_loading_pct": _r(loading, 1), "lines": lines,
                "clean_mix_pct": _r(mix * 100, 1),
            },
            "assets": {
                **{aid: {"available_mw": _r(avail[aid]), "used_mw": _r(used[aid]),
                         "status": p.get(aid).status} for aid in avail},
                **{cid: {"demand_mw": _r(c["base"] + c["firm_in"]),
                         "served_mw": _r(c["after"] - c["shed_nc"] - c["shed_c"])} for cid, c in cons.items()},
            },
            "prices": {"price": _r(price, 1), "sell_price": _r(sell_price, 1),
                       "dsm_deficit_rate": _r(dsm_def, 1), "dsm_surplus_rate": _r(dsm_sur, 1),
                       "carbon_price": _r(act["carbon_price"], 1), "grid_ef": _r(act["grid_ef"], 4),
                       "dr_incentive": _r(act["dr_incentive"], 1), "frequency_hz": _r(freq, 3)},
            "weather": {"temperature_c": _r(act["temperature_c"], 1), "cloudiness": _r(act["cloudiness"], 2)},
            "costs": {k: _r(kd.get(k, 0.0), 2) for k in money_keys},
            "objective": {"components": {k: _r(v, 2) for k, v in comps.items()},
                          "total": _r(default_total, 2), "agent_weights": decision.objective_weights,
                          "agent_total": _r(agent_total, 2)},
            "active_events": [ev.id for ev in self.events if ev.is_active(t)],
            "notes": notes,
            "kpi_delta": dict(kd),
        }
        self.kpis.add(tick)
        if self.keep_history:
            self.history.append(tick)
        self.last_tick = tick

        # ---- 12. advance the clock
        self.pending_shift.pop(t, None)
        self.step += 1
        if self.step % STEPS_PER_DAY == 0:
            for c in p.consumers:
                c.dr_steps_used_today = 0
        self._begin_step()
        return tick

    # ================================================================== reporting
    def event_timeline(self) -> list[dict]:
        """Operator/demo view: includes surprises the agent can't see yet (flagged known_to_agent=False).
        Never feed this to an agent; agents get `observe()["alerts"]`."""
        t = min(self.step, self.total_steps - 1)
        return [{**ev.to_dict(t), "known_to_agent": ev.is_known(t)}
                for ev in sorted(self.events, key=lambda e: e.start_step)]

    def summary(self) -> dict:
        return {
            "scenario": self.scenario.id, "seed": self.seed, "days": self.days,
            "step": self.step, "total_steps": self.total_steps, "done": self.done,
            "kpis": self.kpis.summary(),
            "batteries": {b.id: {"soc": _r(b.soc, 4), "soh": _r(b.soh, 6)} for b in self.portfolio.batteries},
        }


def chart_point(tick: dict) -> dict:
    """Compact per-step record for time-series charts."""
    f, b = tick["flows"], tick["applied"]["batteries"]
    return {
        "step": tick["step"], "time": tick["time"],
        "solar": f["solar_used_mw"], "wind": f["wind_used_mw"],
        "solar_available": f["solar_available_mw"], "wind_available": f["wind_available_mw"],
        "demand": f["demand_served_mw"], "demand_requested": f["demand_requested_mw"],
        "battery_net": f["battery_net_mw"], "grid": f["grid_net_import_mw"],
        "curtailed": _r(f["curtailed_mw"] + f["forced_curtailed_mw"]), "unserved": f["unserved_mw"],
        "price": tick["prices"]["price"], "frequency": tick["prices"]["frequency_hz"],
        "import_limit": f["import_limit_mw"], "clean_mix_pct": f["clean_mix_pct"],
        **{f"soc_{k}": v["soc"] for k, v in b.items()},
        "mode": tick["mode"], "fallback": tick["fallback"],
    }
