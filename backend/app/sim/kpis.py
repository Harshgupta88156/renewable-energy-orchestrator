"""KPIs and the objective function: how every agent is scored, identically.

The problem statement asks to make the optimality criteria explicit. We score a run with a
single money-denominated objective (lower is better) built from transparent components:

    energy          purchases + deviation charges - sales - deviation credits
    carbon          tCO2 imported x carbon price
    degradation     battery wear (₹ per MWh cycled)
    flexibility     DR incentives + shift fees + controlled cuts + tariff lost on curtailed demand
    reliability     value of lost load for shed demand (+ tariff lost)
    maintenance     crew dispatches
    curtailment     wasted clean MWh x shadow price (0 by default, raise it for a "green" policy)

objective = Σ weight_i × component_i. With default weights (all 1, curtailment 0) minimising the
objective is exactly maximising profit, because tariff revenue on total demand is a constant.
An agent (e.g. the LLM planner) can change the weights per situation, and we always report both
its own weighted objective and the default one, so runs stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from .constants import CURTAILMENT_SHADOW_PRICE

DEFAULT_WEIGHTS = {
    "energy": 1.0,
    "carbon": 1.0,
    "degradation": 1.0,
    "flexibility": 1.0,
    "reliability": 1.0,
    "maintenance": 1.0,
    "curtailment": 0.0,
}


@dataclass
class KpiAccumulator:
    battery_nameplate_mwh: float = 160.0
    steps: int = 0
    # energy (MWh)
    demand_requested_mwh: float = 0.0
    demand_served_mwh: float = 0.0
    unserved_noncritical_mwh: float = 0.0
    unserved_critical_mwh: float = 0.0
    dr_mwh: float = 0.0
    shifted_mwh: float = 0.0
    noncritical_cut_mwh: float = 0.0
    renewable_available_mwh: float = 0.0
    renewable_used_mwh: float = 0.0
    curtailed_voluntary_mwh: float = 0.0
    curtailed_forced_mwh: float = 0.0
    import_mwh: float = 0.0
    export_mwh: float = 0.0
    scheduled_buy_mwh: float = 0.0
    scheduled_sell_mwh: float = 0.0
    deviation_import_mwh: float = 0.0
    deviation_export_mwh: float = 0.0
    battery_charge_mwh: float = 0.0
    battery_discharge_mwh: float = 0.0
    emergency_battery_mwh: float = 0.0
    clean_to_demand_mwh: float = 0.0
    low_freq_import_mwh: float = 0.0
    # carbon (t)
    co2_t: float = 0.0
    avoided_co2_t: float = 0.0
    # money (₹)
    purchase_cost: float = 0.0
    sales_revenue: float = 0.0
    dsm_cost: float = 0.0
    dsm_credit: float = 0.0
    carbon_cost: float = 0.0
    degradation_cost: float = 0.0
    dr_cost: float = 0.0
    shift_cost: float = 0.0
    noncritical_cost: float = 0.0
    shed_penalty: float = 0.0
    maintenance_cost: float = 0.0
    tariff_revenue: float = 0.0
    lost_tariff_flex: float = 0.0
    lost_tariff_shed: float = 0.0
    # grid & ops
    max_line_loading_pct: float = 0.0
    steps_at_line_limit: int = 0
    shed_steps: int = 0
    clipped_actions: int = 0
    emergency_actions: int = 0
    fallbacks: int = 0
    decision_latency_ms_total: float = 0.0
    objective_weighted_total: float = 0.0   # using the agent's own weights at each step

    def add(self, tick: dict) -> None:
        k = tick["kpi_delta"]
        for f in fields(self):
            name = f.name
            if name in ("max_line_loading_pct", "battery_nameplate_mwh", "steps"):
                continue
            if name in k:
                setattr(self, name, getattr(self, name) + k[name])
        self.steps += 1
        self.max_line_loading_pct = max(self.max_line_loading_pct, k.get("line_loading_pct", 0.0))

    # ------------------------------------------------------------------ summary
    def components(self) -> dict[str, float]:
        return components_from(self.__dict__)

    def summary(self) -> dict:
        comps = self.components()
        objective = sum(DEFAULT_WEIGHTS[k] * v for k, v in comps.items())
        net_energy_cost = self.purchase_cost + self.dsm_cost - self.sales_revenue - self.dsm_credit
        operating_cost = (net_energy_cost + self.carbon_cost + self.degradation_cost + self.dr_cost
                          + self.shift_cost + self.noncritical_cost + self.shed_penalty
                          + self.maintenance_cost)
        req = max(self.demand_requested_mwh, 1e-9)
        ren = max(self.renewable_available_mwh, 1e-9)
        served = max(self.demand_served_mwh, 1e-9)
        bat_energy = max(self.battery_nameplate_mwh, 1e-9)
        return {
            "steps": self.steps,
            "hours": self.steps / 4,
            "money": {
                "profit": self.tariff_revenue - operating_cost,
                "tariff_revenue": self.tariff_revenue,
                "operating_cost": operating_cost,
                "net_energy_cost": net_energy_cost,
                "purchase_cost": self.purchase_cost,
                "sales_revenue": self.sales_revenue,
                "dsm_cost": self.dsm_cost,
                "dsm_credit": self.dsm_credit,
                "carbon_cost": self.carbon_cost,
                "degradation_cost": self.degradation_cost,
                "dr_cost": self.dr_cost,
                "shift_cost": self.shift_cost,
                "noncritical_cost": self.noncritical_cost,
                "shed_penalty": self.shed_penalty,
                "maintenance_cost": self.maintenance_cost,
            },
            "energy": {
                "demand_requested_mwh": self.demand_requested_mwh,
                "demand_served_mwh": self.demand_served_mwh,
                "unserved_mwh": self.unserved_noncritical_mwh + self.unserved_critical_mwh,
                "unserved_critical_mwh": self.unserved_critical_mwh,
                "renewable_available_mwh": self.renewable_available_mwh,
                "renewable_used_mwh": self.renewable_used_mwh,
                "curtailed_mwh": self.curtailed_voluntary_mwh + self.curtailed_forced_mwh,
                "curtailed_forced_mwh": self.curtailed_forced_mwh,
                "import_mwh": self.import_mwh,
                "export_mwh": self.export_mwh,
                "deviation_mwh": self.deviation_import_mwh + self.deviation_export_mwh,
                "battery_charge_mwh": self.battery_charge_mwh,
                "battery_discharge_mwh": self.battery_discharge_mwh,
                "emergency_battery_mwh": self.emergency_battery_mwh,
                "dr_mwh": self.dr_mwh,
                "shifted_mwh": self.shifted_mwh,
                "noncritical_cut_mwh": self.noncritical_cut_mwh,
            },
            "percent": {
                "reliability_pct": 100 * (req - self.unserved_noncritical_mwh - self.unserved_critical_mwh) / req,
                "renewable_utilization_pct": 100 * self.renewable_used_mwh / ren,
                "clean_share_pct": 100 * self.clean_to_demand_mwh / served,
                "self_sufficiency_pct": 100 * max(0.0, 1 - self.import_mwh / served),
            },
            "carbon": {"co2_t": self.co2_t, "avoided_co2_t": self.avoided_co2_t,
                       "co2_intensity_t_per_mwh": self.co2_t / served},
            "grid": {
                "max_line_loading_pct": self.max_line_loading_pct,
                "steps_at_line_limit": self.steps_at_line_limit,
                "low_freq_import_mwh": self.low_freq_import_mwh,
                "shed_steps": self.shed_steps,
            },
            "battery": {
                "equivalent_full_cycles": (self.battery_charge_mwh + self.battery_discharge_mwh) / 2 / bat_energy,
            },
            "agent": {
                "clipped_actions": self.clipped_actions,
                "emergency_actions": self.emergency_actions,
                "fallbacks": self.fallbacks,
                "avg_latency_ms": self.decision_latency_ms_total / max(self.steps, 1),
            },
            "objective": {
                "weights": DEFAULT_WEIGHTS,
                "components": comps,
                "total": objective,
                "agent_weighted_total": self.objective_weighted_total,
                "lower_is_better": True,
            },
        }


def components_from(k: dict) -> dict[str, float]:
    """Objective components (₹) from either a cumulative accumulator or a single-step delta."""
    g = lambda name: float(k.get(name, 0.0))  # noqa: E731
    return {
        "energy": g("purchase_cost") + g("dsm_cost") - g("sales_revenue") - g("dsm_credit"),
        "carbon": g("carbon_cost"),
        "degradation": g("degradation_cost"),
        "flexibility": g("dr_cost") + g("shift_cost") + g("noncritical_cost") + g("lost_tariff_flex"),
        "reliability": g("shed_penalty") + g("lost_tariff_shed"),
        "maintenance": g("maintenance_cost"),
        "curtailment": (g("curtailed_voluntary_mwh") + g("curtailed_forced_mwh")) * CURTAILMENT_SHADOW_PRICE,
    }


def weighted(components: dict[str, float], weights: dict[str, float] | None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return sum(float(w.get(k, 1.0)) * v for k, v in components.items())
