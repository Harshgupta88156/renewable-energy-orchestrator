"""The baseline must beat 'no intelligence' - the bar every new agent is measured against."""
import pytest

from app.runtime.lab import run_batch
from app.sim.scenarios import SCENARIOS
from tests.test_engine import run


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_rule_based_beats_naive_on_objective(scenario):
    naive, _ = run(scenario, "naive")
    smart, _ = run(scenario, "rule_based")
    assert smart.kpis.summary()["objective"]["total"] < naive.kpis.summary()["objective"]["total"]


def test_storm_preparation_prevents_blackout():
    naive, _ = run("storm_alert", "naive")
    smart, ticks = run("storm_alert", "rule_based")
    assert naive.kpis.summary()["energy"]["unserved_mwh"] > 50
    assert smart.kpis.summary()["energy"]["unserved_mwh"] < 1
    storm = next(e for e in smart.events if e.type == "storm")
    at_storm = ticks[storm.start_step - 1]["applied"]["batteries"]
    assert all(b["soc"] > 0.9 for b in at_storm.values())  # batteries full when the storm hits


def test_rule_based_dispatches_crew_on_fault():
    _, ticks = run("heatwave_peak", "rule_based")
    assert any(m["action"] == "dispatch_inspection" for t in ticks for m in t["applied"]["maintenance"])


def test_lab_paired_comparison():
    res = run_batch("chaos_monkey", ["naive", "rule_based"], runs=4, base_seed=42)
    assert res["summary"]["naive"]["runs"] == 4
    assert res["comparisons"]["rule_based"]["objective_total"]["win_rate"] >= 0.75
