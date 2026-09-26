"""Synthetic but realistic time series (the "true" world before events).

Everything is seeded, so the same seed always produces the same day. That lets us
replay a day exactly and compare agents fairly on identical conditions.

Shapes are modelled on an Indian October day:
  * solar: clear-sky bell shifted by longitude, attenuated by regional clouds and heat
  * wind: diurnal cycle (stronger in the evening/night) + correlated gusts
  * demand: 5 consumer archetypes (flat steel, day-shift cement/textile, data centre, city peak)
  * price: exchange-style curve, cheap in solar hours, expensive in the evening peak
  * grid emission factor: cleaner at midday (solar on the grid), dirtier at night
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assets import Portfolio
from .constants import PRICE_CAP, PRICE_FLOOR, STEPS_PER_DAY, STEPS_PER_HOUR


@dataclass
class WeatherParams:
    cloudiness: float = 0.08          # mean cloud fraction 0..1
    cloud_variability: float = 0.10
    wind_mean_ms: float = 7.5         # hub-height regional mean
    wind_variability: float = 1.2
    temp_mean_c: float = 29.0
    temp_amplitude_c: float = 6.0
    demand_multiplier: float = 1.0
    city_demand_multiplier: float = 1.0
    price_multiplier: float = 1.0
    price_volatility: float = 0.08
    carbon_price: float = 2_000.0     # ₹/tCO2 (shadow price)
    dr_incentive: float = 4_500.0     # ₹/MWh paid to consumers for demand response

    def jittered(self, rng: np.random.Generator) -> "WeatherParams":
        """Randomised variant for Monte Carlo runs (F3)."""
        return WeatherParams(
            cloudiness=float(np.clip(self.cloudiness + rng.normal(0, 0.08), 0, 0.9)),
            cloud_variability=float(np.clip(self.cloud_variability * rng.uniform(0.7, 1.4), 0.02, 0.4)),
            wind_mean_ms=float(np.clip(self.wind_mean_ms + rng.normal(0, 1.0), 3.0, 12.0)),
            wind_variability=float(np.clip(self.wind_variability * rng.uniform(0.7, 1.4), 0.3, 3.0)),
            temp_mean_c=self.temp_mean_c + float(rng.normal(0, 1.5)),
            temp_amplitude_c=self.temp_amplitude_c,
            demand_multiplier=float(self.demand_multiplier * rng.uniform(0.95, 1.05)),
            city_demand_multiplier=float(self.city_demand_multiplier * rng.uniform(0.95, 1.07)),
            price_multiplier=float(self.price_multiplier * rng.uniform(0.85, 1.15)),
            price_volatility=self.price_volatility,
            carbon_price=self.carbon_price,
            dr_incentive=self.dr_incentive,
        )


@dataclass
class BaseSeries:
    n: int
    hours: np.ndarray                     # fractional hour of day per step
    temperature_c: np.ndarray
    cloudiness: np.ndarray                # average cloud fraction (for context)
    solar_cf: dict[str, np.ndarray]       # capacity factor per solar farm (0..1)
    wind_speed: dict[str, np.ndarray]     # m/s per wind farm
    demand: dict[str, np.ndarray]         # MW per consumer
    price: np.ndarray                     # ₹/MWh
    grid_ef: np.ndarray                   # tCO2/MWh of grid imports
    frequency: np.ndarray                 # Hz
    carbon_price: np.ndarray              # ₹/tCO2
    dr_incentive: np.ndarray              # ₹/MWh
    params: WeatherParams = field(default_factory=WeatherParams)


# ---------------------------------------------------------------- helpers

def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Zero-mean AR(1) process with stationary std `sigma` (smooth random wiggles)."""
    x = np.empty(n)
    innov = sigma * np.sqrt(1 - phi**2)
    eps = rng.normal(0.0, 1.0, n)
    x[0] = sigma * eps[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + innov * eps[i]
    return x


def _circ_gauss(h: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    d = np.abs(h - mu)
    d = np.minimum(d, 24 - d)
    return np.exp(-(d**2) / (2 * sigma**2))


def _smoothstep(h: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.clip((h - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def _interp_daily(h: np.ndarray, anchors_h: list[float], anchors_v: list[float]) -> np.ndarray:
    return np.interp(h, anchors_h, anchors_v)


# Exchange-style price curve (₹/MWh): cheap midday solar, expensive evening peak.
_PRICE_H = [0, 4, 6, 8, 9.5, 11, 14, 16, 17.5, 18.5, 20, 22, 23.5, 24]
_PRICE_V = [4200, 3800, 4600, 5800, 4300, 2800, 2500, 3400, 6000, 8600, 9200, 7000, 4800, 4200]
# Grid emission factor (tCO2/MWh): cleaner when national solar is online.
_EF_H = [0, 6, 9, 12, 15, 18, 21, 24]
_EF_V = [0.80, 0.79, 0.70, 0.60, 0.64, 0.78, 0.82, 0.80]


def clear_sky_cf(hours: np.ndarray, lon: float, peak: float = 0.92) -> np.ndarray:
    """Clear-sky capacity factor; sites west of 82.5°E (IST meridian) see the sun later."""
    shift = (82.5 - lon) / 15.0
    sunrise, sunset = 5.75 + shift, 17.45 + shift
    x = (hours - sunrise) / (sunset - sunrise)
    return np.where((x > 0) & (x < 1), peak * np.sin(np.pi * np.clip(x, 0, 1)) ** 1.3, 0.0)


# ---------------------------------------------------------------- generator

def generate_base_series(portfolio: Portfolio, params: WeatherParams, n_days: int,
                         seed: int) -> BaseSeries:
    n = n_days * STEPS_PER_DAY
    rng = np.random.default_rng([seed, 1001])
    steps = np.arange(n)
    hours = (steps % STEPS_PER_DAY) / STEPS_PER_HOUR

    # Temperature: peak ~15:00
    temp = (params.temp_mean_c
            + params.temp_amplitude_c * np.sin(2 * np.pi * (hours - 9) / 24)
            + ar1(rng, n, 0.95, 0.8))

    # Clouds: national + regional + site components
    common = ar1(rng, n, 0.95, params.cloud_variability * 0.6)
    regions = sorted({s.region for s in portfolio.solar})
    regional = {r: ar1(rng, n, 0.92, params.cloud_variability * 0.5) for r in regions}
    solar_cf: dict[str, np.ndarray] = {}
    cloud_stack = []
    for s in portfolio.solar:
        cloud = np.clip(params.cloudiness + common + regional[s.region]
                        + ar1(rng, n, 0.8, 0.03), 0, 1)
        cloud_stack.append(cloud)
        cs = clear_sky_cf(hours, s.lon)
        module_temp = temp + 25 * cs / 0.92
        temp_derate = 1 + s.temp_coeff * np.maximum(0, module_temp - 25)
        solar_cf[s.id] = np.clip(cs * (1 - 0.8 * cloud) * temp_derate, 0, 1)
    cloudiness = np.mean(cloud_stack, axis=0) if cloud_stack else np.zeros(n)

    # Wind: diurnal (stronger late evening) + correlated gusts
    wind_common = ar1(rng, n, 0.96, params.wind_variability)
    wind_speed: dict[str, np.ndarray] = {}
    for w in portfolio.wind:
        diurnal = 1 + 0.18 * np.cos(2 * np.pi * (hours - 21) / 24)
        local = ar1(rng, n, 0.93, params.wind_variability * 0.6)
        wind_speed[w.id] = np.maximum(0.0, params.wind_mean_ms * w.site_factor * diurnal
                                      + wind_common + local)

    # Demand archetypes
    heat = np.clip((temp - params.temp_mean_c) / max(params.temp_amplitude_c, 1e-6), -1, 1.5)
    demand: dict[str, np.ndarray] = {}
    for c in portfolio.consumers:
        noise = 1 + ar1(rng, n, 0.7, 0.02)
        if c.kind == "steel":
            shape = 1 + ar1(rng, n, 0.6, 0.03)
        elif c.kind == "cement":
            shape = 0.72 + 0.28 * (_smoothstep(hours, 7, 9) - _smoothstep(hours, 19, 21))
        elif c.kind == "datacenter":
            shape = 1 + 0.06 * heat
        elif c.kind == "textile":
            shape = 0.35 + 0.65 * (_smoothstep(hours, 5.5, 6.5) - _smoothstep(hours, 21.5, 22.5))
        else:  # city
            shape = (0.55 + 0.17 * _circ_gauss(hours, 9, 1.5)
                     + 0.12 * _circ_gauss(hours, 14, 2.5) * (1 + max(0.0, params.temp_mean_c - 30) / 6)
                     + 0.45 * _circ_gauss(hours, 20.5, 1.8))
            shape = shape * params.city_demand_multiplier
        demand[c.id] = np.maximum(0.0, c.base_mw * shape * noise * params.demand_multiplier)

    # Market
    price_shape = _interp_daily(hours, _PRICE_H, _PRICE_V)
    price = price_shape * params.price_multiplier * (1 + ar1(rng, n, 0.85, params.price_volatility))
    price = np.clip(price, PRICE_FLOOR, PRICE_CAP)
    grid_ef = _interp_daily(hours, _EF_H, _EF_V) * (1 + ar1(rng, n, 0.9, 0.02))
    evening_dip = -0.03 * _circ_gauss(hours, 20, 1.5)
    frequency = np.clip(50.0 + evening_dip + ar1(rng, n, 0.9, 0.015), 49.85, 50.08)

    return BaseSeries(
        n=n, hours=hours, temperature_c=temp, cloudiness=cloudiness,
        solar_cf=solar_cf, wind_speed=wind_speed, demand=demand,
        price=price, grid_ef=grid_ef, frequency=frequency,
        carbon_price=np.full(n, params.carbon_price),
        dr_incentive=np.full(n, params.dr_incentive),
        params=params,
    )
