"""Command line: run scenarios headless, compare agents, run the Monte Carlo lab.

    python -m app.cli scenarios
    python -m app.cli run --scenario storm_alert --agent rule_based --verbose
    python -m app.cli run --scenario storm_alert --csv storm.csv
    python -m app.cli compare --scenario storm_alert
    python -m app.cli compare --all
    python -m app.cli lab --scenario chaos_monkey --runs 50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from .agents import AGENT_CLASSES, create_agent
from .runtime.lab import run_batch
from .sim.engine import Simulator, chart_point
from .sim.scenarios import SCENARIOS, get_scenario

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # ₹ on Windows consoles


def lakh(v: float) -> str:
    return f"₹{v / 1e5:,.2f} L"


def simulate(scenario: str, agent_name: str, seed: int | None, days: int, verbose: bool = False) -> Simulator:
    sim = Simulator(get_scenario(scenario), seed=seed, days=days)
    agent = create_agent(agent_name)
    sim.agent_obj = agent
    last_err = None
    while not sim.done:
        tick = sim.apply_decision(agent.decide_sync(sim.observe()))
        err = getattr(agent, "last_error", None)
        if verbose and err and err != last_err:
            print(f"{'':10}!! LLM error: {err}")
        last_err = err
        if verbose:
            f = tick["flows"]
            socs = " ".join(f"{k}={v['soc'] * 100:3.0f}%" for k, v in tick["applied"]["batteries"].items())
            print(f"{tick['label']}  {tick['mode']:<16} ren {f['renewable_used_mw']:6.1f}  dem {f['demand_served_mw']:6.1f}"
                  f"  grid {f['grid_net_import_mw']:+7.1f}  bat {f['battery_net_mw']:+6.1f}  {socs}"
                  f"  shed {f['unserved_mw']:4.1f}  | {'; '.join(tick['reasons'][:2])}")
            for n in tick["notes"]:
                if n["kind"] == "emergency":
                    print(f"{'':10}!! {n['message']}")
    return sim


def print_kpis(rows: list[tuple[str, dict]]) -> None:
    lines = [
        ("Profit", lambda k: lakh(k["money"]["profit"])),
        ("Objective (lower=better)", lambda k: lakh(k["objective"]["total"])),
        ("Net energy cost", lambda k: lakh(k["money"]["net_energy_cost"])),
        ("Deviation charges", lambda k: lakh(k["money"]["dsm_cost"])),
        ("Unserved energy", lambda k: f"{k['energy']['unserved_mwh']:.1f} MWh"),
        ("Curtailed", lambda k: f"{k['energy']['curtailed_mwh']:.1f} MWh"),
        ("CO2", lambda k: f"{k['carbon']['co2_t']:.0f} t"),
        ("Clean share of demand", lambda k: f"{k['percent']['clean_share_pct']:.1f} %"),
        ("Renewable utilisation", lambda k: f"{k['percent']['renewable_utilization_pct']:.1f} %"),
        ("Battery cycles", lambda k: f"{k['battery']['equivalent_full_cycles']:.2f}"),
        ("Emergency actions", lambda k: f"{k['agent']['emergency_actions']:.0f}"),
    ]
    w = 26
    print(f"{'':{w}}" + "".join(f"{name:>18}" for name, _ in rows))
    for label, fn in lines:
        print(f"{label:{w}}" + "".join(f"{fn(k):>18}" for _, k in rows))


def cmd_scenarios(_args) -> None:
    for s in SCENARIOS.values():
        print(f"{s.id:16} {s.name}\n{'':16} {s.description}\n{'':16} Watch for: {s.watch_for}\n")
    print("Agents:", ", ".join(AGENT_CLASSES))


def cmd_llm_check(args) -> None:
    """One tiny test call to the configured free LLM (checks key, network, tool calling)."""
    import time
    from .agents.llm_providers import make_client, tool_calls
    c = make_client()
    if c is None:
        print("No LLM key found (GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY): planner runs offline.")
        return
    print(f"Provider {c.name}, model {c.model}, url {c.base_url}")
    t0 = time.time()
    try:
        msg, usage = c.chat([{"role": "user", "content": "Call ping with ok=true."}],
                            [{"name": "ping", "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}])
        print(f"OK in {time.time() - t0:.1f}s with model {c.model}, tokens {usage}, tool call: {tool_calls(msg)}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        try:
            print("Models this key can use:", ", ".join(c.available_models()))
            print("Pick one with:  $env:REO_LLM_MODEL=\"<model id>\"")
        except Exception:
            pass


def cmd_run(args) -> None:
    sim = simulate(args.scenario, args.agent, args.seed, args.days, args.verbose)
    print(f"\n{sim.scenario.name} | agent={args.agent} | seed={sim.seed} | days={sim.days}\n")
    print_kpis([(args.agent, sim.kpis.summary())])
    a = sim.agent_obj
    if hasattr(a, "tokens"):
        prov = f"{a.client.name} / {a.client.model}" if a.client else "offline (no LLM key found)"
        print(f"\nPlanner: {prov} | LLM calls {a.calls} | tokens in {a.tokens['in']} out {a.tokens['out']} | "
              f"errors {a.errors} | skipped (free-tier budget) {a.skipped} | twin overrides {a.overrides}")
        if a.last_error:
            print(f"Last LLM error: {a.last_error}")
    if args.csv:
        points = [chart_point(t) for t in sim.history]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(points[0]))
            wr.writeheader()
            wr.writerows(points)
        print(f"\nWrote {len(points)} rows to {args.csv}")


def cmd_compare(args) -> None:
    agents = args.agents.split(",")
    scenarios = list(SCENARIOS) if args.all else [args.scenario]
    for sc in scenarios:
        rows = [(a, simulate(sc, a, args.seed, args.days).kpis.summary()) for a in agents]
        print(f"\n=== {SCENARIOS[sc].name} ({sc}) ===")
        print_kpis(rows)


def cmd_lab(args) -> None:
    agents = args.agents.split(",")
    total = args.runs * len(agents)

    def progress(done: int, tot: int) -> None:
        print(f"\r  {done}/{tot} simulations", end="", flush=True)

    print(f"Scenario lab: {args.scenario}, {args.runs} randomised days x {len(agents)} agents = {total} runs")
    res = run_batch(args.scenario, agents, args.runs, args.days, args.seed, not args.no_randomize, progress)
    print(f"\n  done in {res['elapsed_s']} s\n")
    S = res["summary"]
    w = 28
    print(f"{'':{w}}" + "".join(f"{a:>22}" for a in agents))
    rows = [
        ("Profit mean", lambda a: lakh(S[a]["metrics"]["profit"]["mean"])),
        ("Profit P5 (bad day)", lambda a: lakh(S[a]["metrics"]["profit"]["p5"])),
        ("Objective mean", lambda a: lakh(S[a]["metrics"]["objective_total"]["mean"])),
        ("Days with any load shed", lambda a: f"{S[a]['days_with_load_shed']} / {S[a]['runs']}"),
        ("Unserved total", lambda a: f"{S[a]['total_unserved_mwh']:.1f} MWh"),
        ("Reliability mean", lambda a: f"{S[a]['metrics']['reliability_pct']['mean']:.3f} %"),
        ("Curtailed mean", lambda a: f"{S[a]['metrics']['curtailed_mwh']['mean']:.1f} MWh"),
        ("CO2 mean", lambda a: f"{S[a]['metrics']['co2_t']['mean']:.0f} t"),
        ("Fallbacks", lambda a: f"{S[a]['metrics']['fallbacks']['mean'] * S[a]['runs']:.0f}"),
    ]
    for label, fn in rows:
        print(f"{label:{w}}" + "".join(f"{fn(a):>22}" for a in agents))
    for a, c in res["comparisons"].items():
        print(f"\n{a} vs {c['vs']}: better objective on {c['objective_total']['win_rate'] * 100:.0f}% of days, "
              f"avg profit {lakh(c['profit']['mean_diff'])} per day")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(f"Saved full results to {args.json}")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # ₹ and · print correctly on Windows consoles and files
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Renewable Energy Orchestrator CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios", help="List scenarios and agents").set_defaults(fn=cmd_scenarios)
    sub.add_parser("llm-check", help="Test the free LLM connection").set_defaults(fn=cmd_llm_check)

    r = sub.add_parser("run", help="Run one scenario with one agent")
    r.add_argument("--scenario", default="storm_alert", choices=list(SCENARIOS))
    r.add_argument("--agent", default="rule_based", choices=list(AGENT_CLASSES))
    r.add_argument("--seed", type=int)
    r.add_argument("--days", type=int, default=1)
    r.add_argument("--verbose", "-v", action="store_true", help="print every 15-min decision")
    r.add_argument("--csv", help="export chart data to CSV")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="Same day, several agents, side by side")
    c.add_argument("--scenario", default="storm_alert", choices=list(SCENARIOS))
    c.add_argument("--all", action="store_true", help="every scenario")
    c.add_argument("--agents", default="naive,rule_based")
    c.add_argument("--seed", type=int)
    c.add_argument("--days", type=int, default=1)
    c.set_defaults(fn=cmd_compare)

    lab = sub.add_parser("lab", help="Monte Carlo: many randomised days (F3)")
    lab.add_argument("--scenario", default="chaos_monkey", choices=list(SCENARIOS))
    lab.add_argument("--agents", default="naive,rule_based")
    lab.add_argument("--runs", type=int, default=50)
    lab.add_argument("--days", type=int, default=1)
    lab.add_argument("--seed", type=int, default=1000, help="base seed")
    lab.add_argument("--no-randomize", action="store_true")
    lab.add_argument("--json", help="save full results")
    lab.set_defaults(fn=cmd_lab)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
