"""Events: everything that can change the world while the agent is running.

Covers every example in the problem statement:
    clouds reduce solar          -> cloud_cover
    wind suddenly increases      -> wind_change
    electricity prices spike     -> price_spike
    one battery unavailable      -> asset_fault (fixed faster if the agent dispatches a crew)
    line reaches its capacity    -> line_derate
    industrial demand jumps      -> demand_surge
    weather forecast updated     -> forecast_update (announced ahead of time)
    storm alerts                 -> storm (announced ahead, with a text bulletin)
    maintenance schedules        -> maintenance (agent may delay / reschedule)
    plus frequency_dip, carbon_price_change, dr_incentive_change

Timing model
    start_step      first step the event affects
    duration_steps  None = open-ended (faults until repaired, price changes forever)
    announce_step   when the agent learns about it in advance; None = surprise
                    (the agent only sees it once it has started)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .clock import hhmm, label
from .constants import AUTO_REPAIR_STEPS, PRICE_CAP


class EventType(str, Enum):
    CLOUD_COVER = "cloud_cover"
    WIND_CHANGE = "wind_change"
    PRICE_SPIKE = "price_spike"
    ASSET_FAULT = "asset_fault"
    LINE_DERATE = "line_derate"
    DEMAND_SURGE = "demand_surge"
    FORECAST_UPDATE = "forecast_update"
    STORM = "storm"
    FREQUENCY_DIP = "frequency_dip"
    CARBON_PRICE_CHANGE = "carbon_price_change"
    DR_INCENTIVE_CHANGE = "dr_incentive_change"
    MAINTENANCE = "maintenance"


INF = 10**9
PERSISTENCE_STEPS = 8  # forecasters assume a surprise event lasts ~2 h more


@dataclass
class Window:
    """Time series for steps [a, b) with events applied. Arrays are copies."""
    a: int
    b: int
    solar_cf: dict[str, np.ndarray]
    wind_speed: dict[str, np.ndarray]
    demand: dict[str, np.ndarray]
    price: np.ndarray
    grid_ef: np.ndarray
    frequency: np.ndarray
    carbon_price: np.ndarray
    dr_incentive: np.ndarray
    temperature_c: np.ndarray
    cloudiness: np.ndarray
    available: dict[str, np.ndarray]      # 1/0 availability for solar, wind and battery assets
    line_rating: dict[str, np.ndarray]    # MW limit per tie-line


@dataclass
class Event:
    id: str
    type: str
    start_step: int
    duration_steps: int | None
    params: dict[str, Any] = field(default_factory=dict)
    announce_step: int | None = None
    source: str = "scenario"              # scenario | manual | random
    title: str = ""
    bulletin: str | None = None
    severity: str = "info"                # info | warning | critical
    resolved_step: int | None = None
    cancelled: bool = False

    # ---- timing
    def end_step(self) -> int | None:
        if self.duration_steps is not None:
            return self.start_step + self.duration_steps
        return self.resolved_step

    def expected_end(self) -> int:
        """Best guess of the end, as the agent would see it."""
        end = self.end_step()
        if end is not None:
            return end
        if self.type == EventType.ASSET_FAULT:
            eta = self.params.get("repair_eta_step")
            return int(eta) if eta is not None else self.start_step + AUTO_REPAIR_STEPS
        return INF

    def is_known(self, step: int) -> bool:
        if self.cancelled:
            return False
        return step >= self.start_step or (self.announce_step is not None and step >= self.announce_step)

    def is_active(self, step: int) -> bool:
        if self.cancelled or step < self.start_step:
            return False
        end = self.end_step()
        return end is None or step < end

    def status(self, step: int) -> str:
        if self.cancelled:
            return "cancelled"
        if self.is_active(step):
            return "active"
        end = self.end_step()
        if end is not None and step >= end:
            return "ended"
        if self.is_known(step):
            return "planned" if self.type == EventType.MAINTENANCE else "announced"
        return "scheduled"

    # ---- effect on the world
    def apply(self, w: Window, known_at: int | None = None) -> None:
        """Apply to a window. known_at=None -> the truth. known_at=t -> what an operator at step t
        would assume: announced events with their announced times; surprise events that have
        started are assumed to persist for 2 h (their true end is not known)."""
        if self.cancelled:
            return
        if known_at is None:
            end = self.end_step() or INF
        else:
            true_end = self.end_step()
            if true_end is not None and true_end <= known_at:
                return  # already over (observable)
            if self.announce_step is None and self.duration_steps is not None:
                end = known_at + PERSISTENCE_STEPS
            else:
                end = self.expected_end()
        lo, hi = max(self.start_step, w.a), min(end, w.b)
        if lo >= hi:
            return
        sl = slice(lo - w.a, hi - w.a)
        p, t = self.params, self.type

        if t == EventType.CLOUD_COVER:
            for sid in p.get("sites", []):
                if sid in w.solar_cf:
                    w.solar_cf[sid][sl] *= 1 - float(p.get("drop", 0.5))
        elif t == EventType.FORECAST_UPDATE:
            factor = float(p.get("factor", 1.0))
            target = p.get("target", "solar")
            series = w.solar_cf if target == "solar" else w.wind_speed
            for sid in p.get("sites", []):
                if sid in series:
                    series[sid][sl] *= factor
        elif t == EventType.WIND_CHANGE:
            for sid in p.get("sites", []):
                if sid in w.wind_speed:
                    w.wind_speed[sid][sl] *= float(p.get("factor", 1.0))
        elif t == EventType.STORM:
            storm_wind = float(p.get("storm_wind_ms", 27.0))  # above the 25 m/s cut-out
            for sid in w.wind_speed:
                w.wind_speed[sid][sl] = np.maximum(w.wind_speed[sid][sl] * float(p.get("wind_factor", 2.8)),
                                                   storm_wind)
            for sid in w.solar_cf:
                w.solar_cf[sid][sl] *= 1 - float(p.get("solar_drop", 0.8))
            line_id = p.get("line_id")
            if line_id in w.line_rating:
                w.line_rating[line_id][sl] = np.minimum(w.line_rating[line_id][sl],
                                                        float(p.get("line_rating_mw", 60)))
            w.frequency[sl] += float(p.get("frequency_offset_hz", 0.0))
        elif t == EventType.PRICE_SPIKE:
            w.price[sl] = np.minimum(w.price[sl] * float(p.get("multiplier", 2.0)), PRICE_CAP)
        elif t == EventType.DEMAND_SURGE:
            cid = p.get("consumer_id")
            if cid in w.demand:
                w.demand[cid][sl] = np.maximum(0.0, w.demand[cid][sl] + float(p.get("delta_mw", 0)))
        elif t in (EventType.ASSET_FAULT, EventType.MAINTENANCE):
            aid = p.get("asset_id")
            if aid in w.available:
                w.available[aid][sl] = 0.0
            elif aid in w.line_rating:
                w.line_rating[aid][sl] = 0.0
        elif t == EventType.LINE_DERATE:
            line_id = p.get("line_id")
            if line_id in w.line_rating:
                w.line_rating[line_id][sl] = np.minimum(w.line_rating[line_id][sl],
                                                        float(p.get("rating_mw", 0)))
        elif t == EventType.FREQUENCY_DIP:
            w.frequency[sl] += float(p.get("offset_hz", -0.15))
        elif t == EventType.CARBON_PRICE_CHANGE:
            w.carbon_price[sl] = float(p.get("price", 2000))
        elif t == EventType.DR_INCENTIVE_CHANGE:
            w.dr_incentive[sl] = float(p.get("rate", 4500))

    # ---- presentation
    def to_dict(self, step: int | None = None) -> dict:
        end = self.end_step()
        d = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "severity": self.severity,
            "source": self.source,
            "params": self.params,
            "start_step": self.start_step,
            "end_step": end,
            "expected_end_step": None if self.expected_end() >= INF else self.expected_end(),
            "announce_step": self.announce_step,
            "start": label(self.start_step),
            "end": label(end) if end is not None else None,
            "bulletin": self.bulletin,
        }
        if step is not None:
            d["status"] = self.status(step)
        return d


# ------------------------------------------------------------------ catalog

_CATALOG: dict[EventType, dict] = {
    EventType.CLOUD_COVER: {
        "label": "Cloud cover", "category": "weather", "severity": "warning",
        "description": "Clouds cut solar output at selected farms.",
        "params": {"sites": ["S1", "S2", "S3"], "drop": 0.6},
        "start_in_steps": 0, "duration_steps": 6, "announce_lead_steps": None,
    },
    EventType.WIND_CHANGE: {
        "label": "Wind surge / drop", "category": "weather", "severity": "info",
        "description": "Wind speed multiplied by a factor (>1 surge, <1 lull).",
        "params": {"sites": ["W1", "W2", "W3"], "factor": 1.6},
        "start_in_steps": 0, "duration_steps": 8, "announce_lead_steps": None,
    },
    EventType.PRICE_SPIKE: {
        "label": "Price spike", "category": "market", "severity": "warning",
        "description": "Exchange price multiplied (capped at ₹10,000/MWh).",
        "params": {"multiplier": 2.0},
        "start_in_steps": 0, "duration_steps": 6, "announce_lead_steps": None,
    },
    EventType.ASSET_FAULT: {
        "label": "Asset fault", "category": "asset", "severity": "critical",
        "description": "Asset trips offline until repaired (8 h, or 1 h after a crew is dispatched).",
        "params": {"asset_id": "B2"},
        "start_in_steps": 0, "duration_steps": None, "announce_lead_steps": None,
    },
    EventType.LINE_DERATE: {
        "label": "Line at capacity", "category": "grid", "severity": "warning",
        "description": "Tie-line limited to a lower MW rating (0 = tripped).",
        "params": {"line_id": "L1", "rating_mw": 40},
        "start_in_steps": 0, "duration_steps": 12, "announce_lead_steps": None,
    },
    EventType.DEMAND_SURGE: {
        "label": "Demand surge", "category": "demand", "severity": "warning",
        "description": "A consumer suddenly needs more power.",
        "params": {"consumer_id": "C1", "delta_mw": 25},
        "start_in_steps": 0, "duration_steps": 8, "announce_lead_steps": None,
    },
    EventType.FORECAST_UPDATE: {
        "label": "Forecast update", "category": "weather", "severity": "info",
        "description": "Weather forecast revised: a future change is announced now.",
        "params": {"target": "solar", "sites": ["S1", "S2", "S3", "S4", "S5"], "factor": 0.5},
        "start_in_steps": 8, "duration_steps": 8, "announce_lead_steps": 8,
    },
    EventType.STORM: {
        "label": "Storm alert", "category": "weather", "severity": "critical",
        "description": "Storm announced ahead: turbines hit cut-out, solar collapses, a line is derated.",
        "params": {"wind_factor": 2.8, "storm_wind_ms": 27.0, "solar_drop": 0.8, "line_id": "L1",
                   "line_rating_mw": 80, "frequency_offset_hz": -0.14},
        "start_in_steps": 12, "duration_steps": 10, "announce_lead_steps": 12,
    },
    EventType.FREQUENCY_DIP: {
        "label": "Grid frequency dip", "category": "grid", "severity": "warning",
        "description": "Regional shortfall: frequency falls, deviation charges double.",
        "params": {"offset_hz": -0.15},
        "start_in_steps": 0, "duration_steps": 4, "announce_lead_steps": None,
    },
    EventType.CARBON_PRICE_CHANGE: {
        "label": "Carbon price change", "category": "market", "severity": "info",
        "description": "New carbon price (₹/tCO2) from now on.",
        "params": {"price": 4000},
        "start_in_steps": 0, "duration_steps": None, "announce_lead_steps": None,
    },
    EventType.DR_INCENTIVE_CHANGE: {
        "label": "DR incentive change", "category": "market", "severity": "info",
        "description": "New demand-response incentive (₹/MWh) from now on.",
        "params": {"rate": 3000},
        "start_in_steps": 0, "duration_steps": None, "announce_lead_steps": None,
    },
    EventType.MAINTENANCE: {
        "label": "Maintenance request", "category": "asset", "severity": "info",
        "description": "Planned outage; the agent may delay it within a flexibility window.",
        "params": {"asset_id": "W2", "flexible_steps": 24},
        "start_in_steps": 8, "duration_steps": 8, "announce_lead_steps": 8,
    },
}
# plain-string keys so lookups with "cloud_cover" work
EVENT_CATALOG: dict[str, dict] = {k.value: v for k, v in _CATALOG.items()}


def _pct(x: float) -> str:
    return f"{abs(float(x)) * 100:.0f}%"


def describe(ev: Event, names: dict[str, str]) -> None:
    """Fill in a human-readable title and bulletin text (used by UI and LLM)."""
    p, t = ev.params, ev.type
    s, e = hhmm(ev.start_step), (hhmm(ev.end_step()) if ev.end_step() is not None else "until repaired")
    nm = lambda ids: ", ".join(names.get(i, i) for i in ids)  # noqa: E731

    if t == EventType.CLOUD_COVER:
        ev.title = f"Clouds over {', '.join(p.get('sites', []))}: solar -{_pct(p.get('drop', 0))}"
        ev.bulletin = f"Satellite nowcast: cloud band over {nm(p.get('sites', []))}, {s}-{e} IST."
    elif t == EventType.WIND_CHANGE:
        f = float(p.get("factor", 1))
        ev.title = f"Wind {'surge' if f >= 1 else 'lull'} x{f:.1f} at {', '.join(p.get('sites', []))}"
        ev.bulletin = f"Wind speeds x{f:.1f} at {nm(p.get('sites', []))}, {s}-{e} IST."
    elif t == EventType.PRICE_SPIKE:
        ev.title = f"Price spike x{float(p.get('multiplier', 1)):.1f}"
        ev.bulletin = f"Exchange alert: real-time prices up x{float(p.get('multiplier', 1)):.1f} ({s}-{e})."
    elif t == EventType.ASSET_FAULT:
        aid = p.get("asset_id")
        ev.title = f"FAULT: {aid} offline"
        ev.bulletin = f"ALARM {s}: {names.get(aid, aid)} tripped offline. Dispatch an inspection crew to restore."
    elif t == EventType.LINE_DERATE:
        lid, r = p.get("line_id"), float(p.get("rating_mw", 0))
        ev.title = f"{lid} {'tripped' if r <= 0 else f'limited to {r:.0f} MW'}"
        ev.bulletin = (f"Grid operator notice: {names.get(lid, lid)} "
                       f"{'out of service' if r <= 0 else f'limited to {r:.0f} MW'} {s}-{e} IST.")
    elif t == EventType.DEMAND_SURGE:
        cid = p.get("consumer_id")
        ev.title = f"{cid} demand +{float(p.get('delta_mw', 0)):.0f} MW"
        ev.bulletin = f"{names.get(cid, cid)} requests +{float(p.get('delta_mw', 0)):.0f} MW, {s}-{e}."
    elif t == EventType.FORECAST_UPDATE:
        target, f = p.get("target", "solar"), float(p.get("factor", 1))
        change = f"-{_pct(1 - f)}" if f < 1 else f"+{_pct(f - 1)}"
        ev.title = f"Forecast update: {target} {change} {s}-{e}"
        ev.bulletin = (f"Forecast revision: {target} at {nm(p.get('sites', []))} expected "
                       f"{change} between {s} and {e} IST.")
    elif t == EventType.STORM:
        lid = p.get("line_id")
        ev.title = f"Storm {s}-{e}"
        ev.bulletin = (f"MET WARNING (simulated): severe thunderstorm with squalls 60-80 km/h over "
                       f"Gujarat, Rajasthan and Madhya Pradesh, {s}-{e} IST. Wind turbines likely to hit "
                       f"cut-out, solar output to collapse, {names.get(lid, lid)} may be limited to "
                       f"{float(p.get('line_rating_mw', 0)):.0f} MW.")
    elif t == EventType.FREQUENCY_DIP:
        ev.title = f"Frequency dip {float(p.get('offset_hz', 0)):+.2f} Hz"
        ev.bulletin = "Grid frequency falling due to regional shortfall; deviation charges doubled."
    elif t == EventType.CARBON_PRICE_CHANGE:
        ev.title = f"Carbon price -> ₹{float(p.get('price', 0)):,.0f}/t"
        ev.bulletin = f"Carbon price set to ₹{float(p.get('price', 0)):,.0f}/tCO2 from {s}."
    elif t == EventType.DR_INCENTIVE_CHANGE:
        ev.title = f"DR incentive -> ₹{float(p.get('rate', 0)):,.0f}/MWh"
        ev.bulletin = f"Demand-response incentive set to ₹{float(p.get('rate', 0)):,.0f}/MWh from {s}."
    elif t == EventType.MAINTENANCE:
        aid = p.get("asset_id")
        latest = p.get("flexible_until")
        ev.title = f"Maintenance {aid} {s}-{e}"
        ev.bulletin = (f"Planned maintenance of {names.get(aid, aid)} {s}-{e}"
                       + (f"; can be moved to start as late as {hhmm(int(latest))}." if latest is not None else "."))


def make_event(event_id: str, etype: str, *, start_step: int, duration_steps: int | None,
               params: dict | None = None, announce_step: int | None = None,
               source: str = "scenario", severity: str | None = None,
               names: dict[str, str] | None = None) -> Event:
    if etype not in EVENT_CATALOG:
        raise ValueError(f"Unknown event type '{etype}'. Known: {sorted(EVENT_CATALOG)}")
    cat = EVENT_CATALOG[etype]
    merged = {**cat["params"], **(params or {})}
    if announce_step is not None:
        announce_step = min(announce_step, start_step)
    ev = Event(
        id=event_id, type=EventType(etype).value, start_step=int(start_step),
        duration_steps=None if duration_steps is None else max(1, int(duration_steps)),
        params=merged, announce_step=announce_step, source=source,
        severity=severity or cat["severity"],
    )
    if ev.type == EventType.MAINTENANCE:
        flex = int(merged.pop("flexible_steps", 0) or 0)
        merged.setdefault("flexible_until", ev.start_step + flex)
        merged.setdefault("original_start", ev.start_step)
    describe(ev, names or {})
    return ev
