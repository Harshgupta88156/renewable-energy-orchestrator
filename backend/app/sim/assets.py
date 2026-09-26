"""Physical assets of the utility portfolio.

The default portfolio mirrors the problem statement:
5 solar farms, 3 wind farms, 2 battery systems, 5 industrial/urban consumers,
2 grid tie-lines and access to the electricity market.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np

from .constants import DT_H, EPS


@dataclass
class SolarFarm:
    id: str
    name: str
    capacity_mw: float
    region: str
    lat: float
    lon: float
    temp_coeff: float = -0.0035      # output change per °C of module temperature above 25 °C
    status: str = "online"           # online | maintenance | faulted
    kind: str = "solar"

    def spec(self) -> dict:
        return asdict(self)


@dataclass
class WindFarm:
    id: str
    name: str
    capacity_mw: float
    region: str
    lat: float
    lon: float
    site_factor: float = 1.0         # local wind resource relative to the regional mean
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0         # turbines shut down above this speed (storms!)
    efficiency: float = 0.95
    status: str = "online"
    kind: str = "wind"

    def capacity_factor(self, speed_ms: np.ndarray) -> np.ndarray:
        """Standard cubic power curve between cut-in and rated speed."""
        v = np.asarray(speed_ms, dtype=float)
        ci, r, co = self.cut_in_ms, self.rated_ms, self.cut_out_ms
        cubic = (v**3 - ci**3) / (r**3 - ci**3)
        cf = np.where((v >= ci) & (v < r), cubic, 0.0)
        cf = np.where((v >= r) & (v < co), 1.0, cf)
        return np.clip(cf, 0.0, 1.0) * self.efficiency

    def spec(self) -> dict:
        return asdict(self)


@dataclass
class Battery:
    id: str
    name: str
    energy_mwh: float                # nameplate energy
    power_mw: float                  # max charge / discharge power
    region: str
    lat: float
    lon: float
    eta_charge: float = 0.94
    eta_discharge: float = 0.94      # round trip ~88%
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc: float = 0.50                # state of charge, fraction of effective energy
    soh: float = 1.0                 # state of health (capacity fade)
    degradation_cost_per_mwh: float = 1_200.0   # ₹ per MWh cycled ((charge+discharge)/2)
    fade_per_cycle: float = 0.20 / 6_000        # 20 % fade over 6000 equivalent full cycles
    clean_fraction: float = 0.5      # share of stored energy that came from renewables
    reserve_soc: float = 0.10        # floor set by the agent for emergencies
    status: str = "online"
    kind: str = "battery"

    @property
    def effective_energy_mwh(self) -> float:
        return self.energy_mwh * self.soh

    @property
    def stored_mwh(self) -> float:
        return self.soc * self.effective_energy_mwh

    @property
    def online(self) -> bool:
        return self.status == "online"

    def max_charge_mw(self, already_mw: float = 0.0, dt: float = DT_H) -> float:
        if not self.online:
            return 0.0
        room_mwh = max(0.0, self.soc_max - self.soc) * self.effective_energy_mwh
        by_energy = room_mwh / (self.eta_charge * dt)
        return max(0.0, min(self.power_mw, by_energy) - already_mw)

    def max_discharge_mw(self, floor_soc: float | None = None, already_mw: float = 0.0,
                         dt: float = DT_H) -> float:
        if not self.online:
            return 0.0
        floor = self.soc_min if floor_soc is None else max(self.soc_min, floor_soc)
        avail_mwh = max(0.0, self.soc - floor) * self.effective_energy_mwh
        by_energy = avail_mwh * self.eta_discharge / dt
        return max(0.0, min(self.power_mw, by_energy) - already_mw)

    def apply(self, charge_mw: float, discharge_mw: float, clean_mix: float,
              dt: float = DT_H) -> dict:
        """Update SoC, health and provenance. Returns energy bookkeeping."""
        e_eff = self.effective_energy_mwh
        stored = self.soc * e_eff
        clean_stored = stored * self.clean_fraction
        e_in = charge_mw * self.eta_charge * dt
        e_out = discharge_mw / self.eta_discharge * dt
        clean_out = e_out * self.clean_fraction
        stored_new = max(0.0, stored + e_in - e_out)
        clean_new = max(0.0, clean_stored + e_in * clean_mix - clean_out)
        cycled_mwh = (charge_mw + discharge_mw) * dt / 2
        self.soh = max(0.5, self.soh - cycled_mwh / self.energy_mwh * self.fade_per_cycle)
        self.soc = min(1.0, stored_new / self.effective_energy_mwh) if self.effective_energy_mwh > EPS else 0.0
        self.clean_fraction = clean_new / stored_new if stored_new > EPS else self.clean_fraction
        return {
            "energy_in_mwh": e_in,
            "energy_out_mwh": e_out,
            "clean_out_mwh": clean_out * self.eta_discharge,
            "cycled_mwh": cycled_mwh,
            "degradation_cost": cycled_mwh * self.degradation_cost_per_mwh,
        }

    def spec(self) -> dict:
        d = asdict(self)
        d["effective_energy_mwh"] = self.effective_energy_mwh
        return d


@dataclass
class Consumer:
    id: str
    name: str
    kind: str                        # steel | cement | datacenter | textile | city
    base_mw: float                   # scale of the demand profile (≈ peak)
    critical_fraction: float         # share that must never be cut
    dr_fraction: float               # share that can be reduced via demand response
    shiftable_fraction: float        # share that can be moved to a later time
    tariff_per_mwh: float            # what the consumer pays us
    dr_max_steps_per_day: int        # contractual limit on DR events
    region: str
    lat: float
    lon: float
    dr_steps_used_today: int = 0

    def spec(self) -> dict:
        return asdict(self)


@dataclass
class Line:
    id: str
    name: str
    capacity_mw: float               # nominal thermal limit (both directions)
    rating_mw: float = -1.0          # current limit after derates / trips
    status: str = "online"

    def __post_init__(self):
        if self.rating_mw < 0:
            self.rating_mw = self.capacity_mw

    def spec(self) -> dict:
        return asdict(self)


@dataclass
class Portfolio:
    solar: list[SolarFarm]
    wind: list[WindFarm]
    batteries: list[Battery]
    consumers: list[Consumer]
    lines: list[Line]
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._index = {a.id: a for a in self.all_assets()}

    def all_assets(self):
        return [*self.solar, *self.wind, *self.batteries, *self.consumers, *self.lines]

    def get(self, asset_id: str):
        return self._index.get(asset_id)

    def kind_of(self, asset_id: str) -> str | None:
        a = self.get(asset_id)
        if a is None:
            return None
        if isinstance(a, Consumer):
            return "consumer"
        if isinstance(a, Line):
            return "line"
        return a.kind

    @property
    def solar_capacity_mw(self) -> float:
        return sum(s.capacity_mw for s in self.solar)

    @property
    def wind_capacity_mw(self) -> float:
        return sum(w.capacity_mw for w in self.wind)

    def specs(self) -> dict:
        return {
            "solar": [s.spec() for s in self.solar],
            "wind": [w.spec() for w in self.wind],
            "batteries": [b.spec() for b in self.batteries],
            "consumers": [c.spec() for c in self.consumers],
            "lines": [ln.spec() for ln in self.lines],
            "totals": {
                "solar_capacity_mw": self.solar_capacity_mw,
                "wind_capacity_mw": self.wind_capacity_mw,
                "battery_energy_mwh": sum(b.energy_mwh for b in self.batteries),
                "battery_power_mw": sum(b.power_mw for b in self.batteries),
                "tie_line_capacity_mw": sum(ln.capacity_mw for ln in self.lines),
            },
        }

    def clone(self) -> "Portfolio":
        return copy.deepcopy(self)


def default_portfolio() -> Portfolio:
    """A mid-size Indian utility portfolio (names are fictional, locations indicative)."""
    solar = [
        SolarFarm("S1", "Thar Sun Park", 80, "Rajasthan", 26.29, 73.02),
        SolarFarm("S2", "Bikaner Solar", 60, "Rajasthan", 28.02, 73.31),
        SolarFarm("S3", "Kutch Solar", 70, "Gujarat", 23.73, 69.86),
        SolarFarm("S4", "Neemuch Solar", 50, "Madhya Pradesh", 24.47, 74.87),
        SolarFarm("S5", "Rewa Solar", 40, "Madhya Pradesh", 24.53, 81.30),
    ]
    wind = [
        WindFarm("W1", "Kutch Wind", 80, "Gujarat", 23.20, 69.50, site_factor=1.05),
        WindFarm("W2", "Muppandal Wind", 60, "Tamil Nadu", 8.26, 77.55, site_factor=1.10),
        WindFarm("W3", "Dewas Wind", 50, "Madhya Pradesh", 22.97, 76.05, site_factor=0.90),
    ]
    batteries = [
        Battery("B1", "BESS Indore", 100, 50, "Madhya Pradesh", 22.72, 75.86),
        Battery("B2", "BESS Jodhpur", 60, 30, "Rajasthan", 26.24, 73.02),
    ]
    consumers = [
        Consumer("C1", "Pithampur Steel", "steel", 70, 0.80, 0.10, 0.10, 7_500, 8,
                 "Madhya Pradesh", 22.61, 75.68),
        Consumer("C2", "Satna Cement", "cement", 40, 0.50, 0.15, 0.30, 7_200, 12,
                 "Madhya Pradesh", 24.58, 80.83),
        Consumer("C3", "Cloud Data Centre", "datacenter", 30, 1.00, 0.00, 0.00, 8_200, 0,
                 "Madhya Pradesh", 22.75, 75.90),
        Consumer("C4", "Textile Mills", "textile", 25, 0.55, 0.20, 0.20, 7_000, 12,
                 "Madhya Pradesh", 22.53, 75.76),
        Consumer("C5", "Indore City Feeder", "city", 110, 0.70, 0.10, 0.05, 6_000, 8,
                 "Madhya Pradesh", 22.72, 75.86),
    ]
    lines = [
        Line("L1", "Tie-line A (400 kV)", 120),
        Line("L2", "Tie-line B (220 kV)", 100),
    ]
    return Portfolio(solar, wind, batteries, consumers, lines)
