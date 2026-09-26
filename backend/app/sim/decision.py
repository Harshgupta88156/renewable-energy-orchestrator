"""The action vocabulary: what any agent (rules, optimizer, LLM) can tell the simulator to do.

Every field maps to an action family in the problem statement:
    battery           -> charge / discharge / reserve capacity for emergencies
    buy_mw, sell_mw   -> buy / sell electricity (delaying a sale = charge now, sell later)
    curtail_mw        -> curtail wind / solar (asset id, or "solar" / "wind" for a total)
    demand_response_mw, reduce_noncritical_mw, shift_load -> demand management
    maintenance       -> delay / schedule maintenance, dispatch inspection teams
    mode, reasons, objective_weights -> explainability: what the agent optimised for and why

`Decision.from_dict` is deliberately forgiving so JSON from an LLM or another service can be
passed straight in; the simulator then clips anything physically impossible and reports it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return default
    return v if v == v and abs(v) != float("inf") else default  # NaN / inf guard


@dataclass
class BatteryCommand:
    action: str = "idle"              # charge | discharge | idle
    mw: float = 0.0
    reserve_soc: float | None = None  # emergency floor the agent wants to keep (0..1)

    @classmethod
    def from_any(cls, d: Any) -> "BatteryCommand":
        if isinstance(d, BatteryCommand):
            return d
        if isinstance(d, (int, float)) and not isinstance(d, bool):  # signed shorthand: +charge / -discharge
            v = _f(d)
            return cls("charge" if v > 0 else "discharge" if v < 0 else "idle", abs(v))
        if not isinstance(d, dict):
            return cls()
        action = str(d.get("action", "idle")).lower()
        if action not in ("charge", "discharge", "idle"):
            action = "idle"
        reserve = d.get("reserve_soc")
        return cls(action, max(0.0, _f(d.get("mw"))),
                   None if reserve is None else min(1.0, max(0.0, _f(reserve))))


@dataclass
class ShiftCommand:
    consumer_id: str
    mw: float
    delay_steps: int = 8


@dataclass
class MaintenanceCommand:
    action: str                        # delay | schedule | dispatch_inspection
    maintenance_id: str | None = None  # event id of the maintenance window
    asset_id: str | None = None        # for dispatch_inspection
    delay_steps: int | None = None
    start_step: int | None = None


@dataclass
class Decision:
    battery: dict[str, BatteryCommand] = field(default_factory=dict)
    buy_mw: float = 0.0
    sell_mw: float = 0.0
    curtail_mw: dict[str, float] = field(default_factory=dict)
    demand_response_mw: dict[str, float] = field(default_factory=dict)
    reduce_noncritical_mw: dict[str, float] = field(default_factory=dict)
    shift_load: list[ShiftCommand] = field(default_factory=list)
    maintenance: list[MaintenanceCommand] = field(default_factory=list)
    mode: str = "normal"
    reasons: list[str] = field(default_factory=list)
    objective_weights: dict[str, float] | None = None
    agent: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Decision":
        d = d if isinstance(d, dict) else {}
        as_dict = lambda x: x if isinstance(x, dict) else {}  # noqa: E731
        as_list = lambda x: x if isinstance(x, list) else []  # noqa: E731
        num_map = lambda m: {str(k): max(0.0, _f(v)) for k, v in as_dict(m).items()}  # noqa: E731
        shifts = []
        for s in as_list(d.get("shift_load")):
            if isinstance(s, dict) and s.get("consumer_id"):
                shifts.append(ShiftCommand(str(s["consumer_id"]), max(0.0, _f(s.get("mw"))),
                                           int(_f(s.get("delay_steps"), 8))))
        maint = []
        for m in as_list(d.get("maintenance")):
            if isinstance(m, dict) and m.get("action"):
                maint.append(MaintenanceCommand(
                    action=str(m["action"]), maintenance_id=m.get("maintenance_id"),
                    asset_id=m.get("asset_id"),
                    delay_steps=None if m.get("delay_steps") is None else int(_f(m["delay_steps"])),
                    start_step=None if m.get("start_step") is None else int(_f(m["start_step"])),
                ))
        weights = None
        if isinstance(d.get("objective_weights"), dict):
            weights = {str(k): _f(v, -1.0) for k, v in d["objective_weights"].items()}
            weights = {k: v for k, v in weights.items() if 0.0 <= v <= 100.0} or None
        reasons = d.get("reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        reasons = as_list(reasons)
        return cls(
            battery={str(k): BatteryCommand.from_any(v) for k, v in as_dict(d.get("battery")).items()},
            buy_mw=max(0.0, _f(d.get("buy_mw"))),
            sell_mw=max(0.0, _f(d.get("sell_mw"))),
            curtail_mw=num_map(d.get("curtail_mw")),
            demand_response_mw=num_map(d.get("demand_response_mw")),
            reduce_noncritical_mw=num_map(d.get("reduce_noncritical_mw")),
            shift_load=shifts,
            maintenance=maint,
            mode=str(d.get("mode", "normal")),
            reasons=[str(r) for r in reasons],
            objective_weights=weights,
            agent=str(d.get("agent", "")),
            meta=dict(as_dict(d.get("meta"))),
        )

    @classmethod
    def idle(cls, agent: str = "idle", reason: str = "No action") -> "Decision":
        return cls(agent=agent, reasons=[reason])
