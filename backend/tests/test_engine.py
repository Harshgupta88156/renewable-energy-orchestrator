"""Physics, accounting and event invariants of the simulator."""
import math

import pytest

from app.agents import create_agent
from app.sim.clock import at
from app.sim.decision import BatteryCommand, Decision, MaintenanceCommand, ShiftCommand
from app.sim.engine import Simulator
from app.sim.scenarios import SCENARIOS


def run(scenario: str, agent: str = "rule_based", seed=None, days: int = 1, randomize=False):
    sim = Simulator(SCENARIOS[scenario], seed=seed, days=days, randomize=randomize)
    ag = create_agent(agent)
    ticks = []
    while not sim.done:
        ticks.append(sim.apply_decision(ag.decide_sync(sim.observe())))
    return sim, ticks


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("agent", ["naive", "rule_based"])
def test_energy_balance_and_limits(scenario, agent):
    sim, ticks = run(scenario, agent)
    for t in ticks:
        k = t["kpi_delta"]
        supply = k["renewable_used_mwh"] + k["battery_discharge_mwh"] + k["import_mwh"]
        use = k["demand_served_mwh"] + k["battery_charge_mwh"] + k["export_mwh"]
        assert supply == pytest.approx(use, abs=1e-6), t["label"]
        f = t["flows"]
        assert abs(f["grid_net_import_mw"]) <= f["import_limit_mw"] + 1e-3, t["label"]
        for b in t["applied"]["batteries"].values():
            assert 0.10 - 1e-6 <= b["soc"] <= 0.95 + 1e-6
        for v in k.values():
            assert math.isfinite(v)
    s = sim.kpis.summary()
    assert s["energy"]["demand_served_mwh"] <= s["energy"]["demand_requested_mwh"] + 1e-6


def test_multi_day_and_randomized_runs_are_consistent():
    sim, ticks = run("chaos_monkey", days=2, randomize=True, seed=7)
    assert len(ticks) == 192
    assert sim.kpis.summary()["steps"] == 192


def test_same_seed_is_deterministic_and_seed_matters():
    a, _ = run("storm_alert", seed=5)
    b, _ = run("storm_alert", seed=5)
    c, _ = run("storm_alert", seed=6)
    assert a.kpis.summary() == b.kpis.summary()
    assert a.kpis.summary()["money"]["profit"] != c.kpis.summary()["money"]["profit"]


def test_surprise_event_invisible_until_it_starts():
    sim = Simulator(SCENARIOS["monsoon_clouds"])  # surprise cloud cover at 12:15
    cloud = next(e for e in sim.events if e.type == "cloud_cover")
    while sim.step < cloud.start_step:
        assert cloud.id not in {a["id"] for a in sim.observe()["alerts"]}
        sim.apply_decision(None)
    assert cloud.id in {a["id"] for a in sim.observe()["alerts"]}


def test_forecast_update_is_visible_ahead_of_time():
    sim = Simulator(SCENARIOS["monsoon_clouds"])  # announced 11:00 for 15:00-17:00, solar x0.5
    upd = next(e for e in sim.events if e.type == "forecast_update")
    while sim.step < upd.announce_step - 1:
        sim.apply_decision(None)
    before = sim.observe()
    sim.apply_decision(None)
    after = sim.observe()
    assert upd.id not in {a["id"] for a in before["alerts"]}
    assert upd.id in {a["id"] for a in after["alerts"]}
    i = upd.start_step - after["forecast"]["start_step"]
    j = upd.start_step - before["forecast"]["start_step"]
    assert after["forecast"]["solar_total_mw"]["p50"][i] < 0.8 * before["forecast"]["solar_total_mw"]["p50"][j]


def test_battery_commands_are_clipped_and_reported():
    sim = Simulator(SCENARIOS["normal_day"])
    tick = sim.apply_decision(Decision(battery={"B1": BatteryCommand("discharge", 999)}))
    assert tick["applied"]["batteries"]["B1"]["discharge_mw"] <= 50 + 1e-9
    assert any(n["kind"] == "clipped" and n["asset"] == "B1" for n in tick["notes"])


def test_reserve_floor_blocks_agent_but_not_emergency():
    sim = Simulator(SCENARIOS["normal_day"])
    tick = sim.apply_decision(Decision(battery={"B1": BatteryCommand("discharge", 20, reserve_soc=0.9)}))
    assert tick["applied"]["batteries"]["B1"]["discharge_mw"] == 0  # soc 0.5 < reserve 0.9


def test_fault_crew_dispatch_repairs_in_one_hour():
    sim = Simulator(SCENARIOS["normal_day"])
    sim.inject_event("asset_fault", {"asset_id": "B1"})
    assert sim.portfolio.get("B1").status == "faulted"
    sim.apply_decision(Decision(maintenance=[MaintenanceCommand("dispatch_inspection", asset_id="B1")]))
    for _ in range(2):  # crew sent at step 0 -> still down at steps 1..3
        assert sim.portfolio.get("B1").status == "faulted"
        sim.apply_decision(None)
    assert sim.portfolio.get("B1").status == "faulted"
    sim.apply_decision(None)
    assert sim.step == 4 and sim.portfolio.get("B1").status == "online"
    assert sim.kpis.maintenance_cost > 0


def test_fault_without_crew_lasts_eight_hours():
    sim = Simulator(SCENARIOS["normal_day"])
    sim.inject_event("asset_fault", {"asset_id": "W1"})
    for _ in range(31):
        sim.apply_decision(None)
        assert sim.portfolio.get("W1").status == "faulted"
    sim.apply_decision(None)
    assert sim.portfolio.get("W1").status == "online"


def test_maintenance_can_be_moved_only_inside_its_window():
    sim = Simulator(SCENARIOS["normal_day"])  # W1 maintenance 18:00, flexible until 21:00
    mnt = next(e for e in sim.events if e.type == "maintenance")
    sim.apply_decision(Decision(maintenance=[MaintenanceCommand("delay", maintenance_id=mnt.id, delay_steps=200)]))
    assert mnt.start_step == mnt.params["flexible_until"] == at(21)


def test_line_trip_limits_imports_and_forces_shedding():
    sim = Simulator(SCENARIOS["normal_day"])
    sim.inject_event("line_derate", {"line_id": "L1", "rating_mw": 0})
    sim.inject_event("line_derate", {"line_id": "L2", "rating_mw": 0})
    tick = sim.apply_decision(Decision(buy_mw=150))
    assert tick["flows"]["grid_net_import_mw"] == 0
    assert tick["flows"]["unserved_mw"] > 0
    assert any("LOAD SHED" in n["message"] for n in tick["notes"])


def test_shifted_load_comes_back_later():
    sim = Simulator(SCENARIOS["normal_day"])
    t0 = sim.apply_decision(Decision(shift_load=[ShiftCommand("C2", 5, delay_steps=4)]))
    assert t0["flows"]["shifted_out_mw"] == pytest.approx(5, abs=1e-6)
    for _ in range(3):
        sim.apply_decision(None)
    t4 = sim.apply_decision(None)
    assert t4["flows"]["shifted_in_mw"] == pytest.approx(5, abs=1e-6)


def test_storm_turbines_cut_out():
    sim = Simulator(SCENARIOS["storm_alert"])
    storm = next(e for e in sim.events if e.type == "storm")
    while sim.step < storm.start_step:
        sim.apply_decision(None)
    assert sum(sim.actual["wind_mw"].values()) == 0


def test_decision_parsing_is_forgiving():
    d = Decision.from_dict({
        "battery": {"B1": {"action": "DISCHARGE", "mw": "30", "reserve_soc": 0.4}, "B2": -12},
        "buy_mw": None, "sell_mw": "abc", "curtail_mw": {"solar": 5},
        "shift_load": [{"consumer_id": "C2", "mw": 4}], "reasons": "one reason",
    })
    assert d.battery["B1"].action == "discharge" and d.battery["B1"].mw == 30
    assert d.battery["B2"].action == "discharge" and d.battery["B2"].mw == 12
    assert d.buy_mw == 0 and d.sell_mw == 0 and d.reasons == ["one reason"]
    assert d.shift_load[0].delay_steps == 8


# ---------------------------------------------------------------- regressions (independent review)

def test_shift_past_end_of_run_is_rejected():
    sim = Simulator(SCENARIOS["normal_day"])
    while sim.step < 94:
        sim.apply_decision(None)
    tick = sim.apply_decision(Decision(shift_load=[ShiftCommand("C2", 5, delay_steps=8)]))
    assert tick["flows"]["shifted_out_mw"] == 0
    assert any("past the end" in n["message"] for n in tick["notes"])


def test_protection_undoes_agent_curtailment_before_shedding():
    sim = Simulator(SCENARIOS["normal_day"])
    while sim.step < at(12, 30):
        sim.apply_decision(None)
    sim.inject_event("line_derate", {"line_id": "L2", "rating_mw": 0})
    sim.inject_event("line_derate", {"line_id": "L1", "rating_mw": 40})
    tick = sim.apply_decision(Decision(curtail_mw={"solar": 150}))
    assert tick["flows"]["unserved_mw"] == 0
    assert any("curtailment" in n["message"] and "undone" in n["message"] for n in tick["notes"])


@pytest.mark.parametrize("etype,params", [
    ("storm", {"solar_drop": "heavy"}),
    ("storm", {"frequency_offset_hz": None}),
    ("line_derate", {"line_id": "L1", "rating_mw": -300}),
    ("line_derate", {"line_id": "L1", "rating_mw": 500}),
    ("cloud_cover", {"drop": 1.8}),
    ("cloud_cover", {"sites": []}),
    ("demand_surge", {"consumer_id": "C9"}),
])
def test_bad_event_parameters_are_rejected_without_side_effects(etype, params):
    sim = Simulator(SCENARIOS["normal_day"])
    n = len(sim.events)
    with pytest.raises(ValueError):
        sim.inject_event(etype, params)
    assert len(sim.events) == n
    sim.observe()
    sim.apply_decision(None)  # the run still works


def test_bad_objective_weights_are_ignored_not_fatal():
    sim = Simulator(SCENARIOS["normal_day"])
    tick = sim.apply_decision({"battery": {"B1": {"action": "discharge", "mw": 10}},
                               "objective_weights": {"energy": "high"}})
    assert tick["step"] == 0 and sim.step == 1


def test_rule_based_moves_maintenance_to_a_low_wind_window():
    _, ticks = run("normal_day", "rule_based")
    moves = [m for t in ticks for m in t["applied"]["maintenance"] if m["action"] == "schedule"]
    assert moves, "expected the W3 maintenance window to be moved"


def test_surprise_event_end_is_not_leaked_to_forecast():
    sim = Simulator(SCENARIOS["normal_day"])
    while sim.step < at(11):
        sim.apply_decision(None)
    sim.inject_event("cloud_cover", {"sites": ["S1", "S2", "S3", "S4", "S5"], "drop": 0.9}, duration_steps=2)
    fc = sim.observe()["forecast"]["solar_total_mw"]["p50"]
    # the true event ends after 30 min, but the forecaster assumes it persists ~2 h
    assert fc[3] < 0.5 * fc[12]


def test_crew_is_not_sent_when_auto_repair_is_sooner():
    sim = Simulator(SCENARIOS["normal_day"])
    sim.inject_event("asset_fault", {"asset_id": "W1"})
    for _ in range(30):
        sim.apply_decision(None)
    tick = sim.apply_decision(Decision(maintenance=[MaintenanceCommand("dispatch_inspection", asset_id="W1")]))
    assert tick["kpi_delta"].get("maintenance_cost", 0) == 0
