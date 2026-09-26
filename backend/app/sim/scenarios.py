"""Preset scenario days (for the live demo) and the random chaos generator (for F3 Monte Carlo).

Each scripted scenario tells one story a judge can follow; `chaos_monkey` throws random events
from every category so the scenario lab can prove reliability over many simulated days.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .clock import at
from .constants import STEPS_PER_DAY
from .profiles import WeatherParams

SOLAR_IDS = ["S1", "S2", "S3", "S4", "S5"]
WIND_IDS = ["W1", "W2", "W3"]
BATTERY_IDS = ["B1", "B2"]
LINE_CAPS = {"L1": 120, "L2": 100}
FLEX_CONSUMERS = ["C1", "C2", "C4", "C5"]


def ev(etype: str, start: int, duration: int | None, announce: int | None = None, **params) -> dict:
    return {"type": etype, "start_step": start, "duration_steps": duration,
            "announce_step": announce, "params": params}


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    watch_for: str
    seed: int
    weather: WeatherParams
    script: Callable[[], list[dict]] = field(default=lambda: [])
    start_date: str = "2026-10-15"
    initial_soc: dict[str, float] = field(default_factory=lambda: {"B1": 0.5, "B2": 0.5})
    chaos_rate: float = 0.0          # multiplier on random-event rates for day 1

    def build_events(self, rng: np.random.Generator, days: int, randomize: bool) -> list[dict]:
        specs = [dict(s) for s in self.script()]
        if randomize:
            for s in specs:
                shift = int(rng.integers(-3, 4))
                s["start_step"] = max(0, s["start_step"] + shift)
                if s.get("announce_step") is not None:
                    s["announce_step"] = max(0, min(s["announce_step"] + shift, s["start_step"]))
        for day in range(days):
            rate = self.chaos_rate if day == 0 else max(self.chaos_rate, 0.5)
            if randomize:
                rate += 0.35
            if rate > 0:
                specs += random_events(rng, day, rate)
        return specs

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description,
                "watch_for": self.watch_for, "seed": self.seed, "start_date": self.start_date}


# ------------------------------------------------------------------ random chaos

RANDOM_RATES = {  # expected events per day at rate 1.0
    "cloud_cover": 2.0, "wind_change": 1.0, "price_spike": 1.0, "asset_fault": 0.5,
    "line_derate": 0.6, "demand_surge": 1.2, "forecast_update": 1.0, "storm": 0.25,
    "frequency_dip": 0.8,
}


def random_events(rng: np.random.Generator, day: int, rate: float) -> list[dict]:
    d0 = day * STEPS_PER_DAY
    out: list[dict] = []
    pick = lambda items, k: [str(x) for x in rng.choice(items, size=k, replace=False)]  # noqa: E731
    for etype, lam in RANDOM_RATES.items():
        for _ in range(int(rng.poisson(lam * rate))):
            if etype == "cloud_cover":
                out.append(ev(etype, d0 + int(rng.integers(at(7), at(17))), int(rng.integers(2, 11)),
                              sites=pick(SOLAR_IDS, int(rng.integers(2, 5))),
                              drop=round(float(rng.uniform(0.3, 0.85)), 2)))
            elif etype == "wind_change":
                factor = rng.uniform(0.3, 0.6) if rng.random() < 0.5 else rng.uniform(1.3, 1.9)
                out.append(ev(etype, d0 + int(rng.integers(0, STEPS_PER_DAY - 4)), int(rng.integers(4, 17)),
                              sites=pick(WIND_IDS, int(rng.integers(1, 4))), factor=round(float(factor), 2)))
            elif etype == "price_spike":
                start = rng.integers(at(17), at(22)) if rng.random() < 0.6 else rng.integers(0, STEPS_PER_DAY - 4)
                out.append(ev(etype, d0 + int(start), int(rng.integers(2, 9)),
                              multiplier=round(float(rng.uniform(1.3, 2.2)), 2)))
            elif etype == "asset_fault":
                asset = str(rng.choice(BATTERY_IDS + SOLAR_IDS + WIND_IDS))
                out.append(ev(etype, d0 + int(rng.integers(0, STEPS_PER_DAY - 4)), None, asset_id=asset))
            elif etype == "line_derate":
                line = str(rng.choice(list(LINE_CAPS)))
                out.append(ev(etype, d0 + int(rng.integers(0, STEPS_PER_DAY - 4)), int(rng.integers(4, 17)),
                              line_id=line, rating_mw=round(float(rng.uniform(0, 0.5) * LINE_CAPS[line]))))
            elif etype == "demand_surge":
                out.append(ev(etype, d0 + int(rng.integers(0, STEPS_PER_DAY - 4)), int(rng.integers(4, 13)),
                              consumer_id=str(rng.choice(FLEX_CONSUMERS)),
                              delta_mw=round(float(rng.uniform(10, 35)))))
            elif etype == "forecast_update":
                lead = int(rng.integers(4, 13))
                start = d0 + int(rng.integers(at(9), at(15)))
                if rng.random() < 0.6:
                    params = {"target": "solar", "sites": SOLAR_IDS, "factor": round(float(rng.uniform(0.4, 0.8)), 2)}
                else:
                    params = {"target": "wind", "sites": WIND_IDS, "factor": round(float(rng.uniform(0.5, 1.6)), 2)}
                out.append(ev(etype, start, int(rng.integers(4, 13)), announce=max(d0, start - lead), **params))
            elif etype == "storm":
                start = d0 + int(rng.integers(at(14), at(20)))
                line = str(rng.choice(list(LINE_CAPS)))
                out.append(ev(etype, start, int(rng.integers(6, 13)),
                              announce=max(d0, start - int(rng.integers(8, 21))),
                              wind_factor=round(float(rng.uniform(2.3, 3.0)), 2), storm_wind_ms=27.0,
                              solar_drop=round(float(rng.uniform(0.6, 0.9)), 2),
                              line_id=line, line_rating_mw=round(float(rng.uniform(30, 70))),
                              frequency_offset_hz=round(float(rng.uniform(-0.18, -0.08)), 3)))
            elif etype == "frequency_dip":
                out.append(ev(etype, d0 + int(rng.integers(0, STEPS_PER_DAY - 4)), int(rng.integers(2, 7)),
                              offset_hz=round(float(rng.uniform(-0.2, -0.1)), 3)))
    return out


# ------------------------------------------------------------------ scripted days

def _normal_day() -> list[dict]:
    return [
        ev("maintenance", at(18), 12, announce=0, asset_id="W1", flexible_steps=at(21) - at(18)),
        ev("demand_surge", at(11, 30), 4, consumer_id="C1", delta_mw=20),
        ev("price_spike", at(19), 4, multiplier=1.4),
    ]


def _monsoon() -> list[dict]:
    return [
        ev("wind_change", at(2), 12, sites=WIND_IDS, factor=1.5),
        ev("forecast_update", at(15), 8, announce=at(11), target="solar", sites=SOLAR_IDS, factor=0.5),
        ev("cloud_cover", at(12, 15), 6, sites=["S1", "S2", "S3"], drop=0.6),
    ]


def _heatwave() -> list[dict]:
    return [
        ev("dr_incentive_change", at(16), None, announce=at(12), rate=3500),
        ev("demand_surge", at(14), 12, consumer_id="C5", delta_mw=25),
        ev("asset_fault", at(17, 30), None, asset_id="B2"),
        ev("price_spike", at(19, 30), 6, multiplier=1.5),
    ]


def _storm() -> list[dict]:
    return [
        ev("storm", at(16, 30), 10, announce=at(12, 30), wind_factor=2.8, storm_wind_ms=27.0,
           solar_drop=0.8, line_id="L1", line_rating_mw=80, frequency_offset_hz=-0.14),
    ]


def _congestion() -> list[dict]:
    return [
        ev("line_derate", at(10), 20, line_id="L2", rating_mw=0),
        ev("line_derate", at(11), 12, announce=at(8), line_id="L1", rating_mw=70),
        ev("wind_change", at(12), 8, sites=WIND_IDS, factor=1.4),
        ev("carbon_price_change", at(16), None, announce=at(9), price=4000),
    ]


SCENARIOS: dict[str, Scenario] = {s.id: s for s in [
    Scenario(
        "normal_day", "Clear October day",
        "Sunny, steady wind. Kutch Wind maintenance is planned for 18:00 (windy evening peak), plus a steel-plant surge and an evening price spike.",
        "Battery arbitrage (charge in cheap solar hours, discharge in the evening peak) and moving the maintenance to a calm window.",
        seed=11, weather=WeatherParams(cloudiness=0.05, cloud_variability=0.06, wind_mean_ms=7.5),
        script=_normal_day,
    ),
    Scenario(
        "monsoon_clouds", "Monsoon clouds",
        "Patchy clouds and strong wind. A surprise cloud band at 12:15 and a forecast revision announcing more clouds at 15:00.",
        "Covering sudden solar drops, and pre-charging after the forecast update.",
        seed=22, weather=WeatherParams(cloudiness=0.35, cloud_variability=0.18, wind_mean_ms=8.5),
        script=_monsoon,
    ),
    Scenario(
        "heatwave_peak", "Heatwave evening peak",
        "Hot day, weak wind, AC load surge. Battery B2 trips at 17:30 right before a price spike.",
        "Dispatching an inspection crew, demand response when the grid is expensive.",
        seed=33, weather=WeatherParams(cloudiness=0.04, cloud_variability=0.05, wind_mean_ms=5.5,
                                       temp_mean_c=36, temp_amplitude_c=7, city_demand_multiplier=1.05,
                                       price_multiplier=1.12),
        script=_heatwave,
    ),
    Scenario(
        "storm_alert", "Evening storm",
        "A storm warning arrives at 12:30 for 16:30-19:00: turbines hit cut-out, solar collapses, Tie-line A is limited to 80 MW.",
        "Preparation: full batteries and flexible load lined up before the storm, so no load is shed.",
        seed=44, weather=WeatherParams(cloudiness=0.12, cloud_variability=0.08, wind_mean_ms=7.0),
        script=_storm,
    ),
    Scenario(
        "grid_congestion", "Congested grid",
        "Very sunny and windy while Tie-line B trips and Tie-line A is limited: nowhere to export the surplus.",
        "Storing surplus instead of curtailing, and choosing what to curtail when storage is full.",
        seed=55, weather=WeatherParams(cloudiness=0.03, cloud_variability=0.04, wind_mean_ms=9.5),
        script=_congestion,
    ),
    Scenario(
        "chaos_monkey", "Random chaos",
        "Random events from every category (clouds, faults, spikes, storms, line limits...). Different every seed.",
        "Robustness: use it in the scenario lab to run many days and compare agents.",
        seed=66, weather=WeatherParams(cloudiness=0.15, cloud_variability=0.12, wind_mean_ms=7.5),
        chaos_rate=1.0,
    ),
]}


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Known: {sorted(SCENARIOS)}")
    return SCENARIOS[scenario_id]
