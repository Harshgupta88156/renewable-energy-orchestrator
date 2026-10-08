"""Rule-based operator: a solid heuristic dispatcher.

Two jobs:
  1. Baseline: what a careful human operator with a spreadsheet would do.
  2. Safety net: the runtime falls back to it whenever a smarter agent times out or errors.

Policy, in priority order (reliability > cost > carbon):
  * Maintenance: send a crew to any faulted asset; move flexible maintenance to the window
    with the least forecast generation.
  * Risk look-ahead (6 h): pessimistic forecast (P10 renewables, P90 demand) against known
    grid limits. If a shortfall beyond the tie-line capacity is coming, hold that energy
    as battery reserve and pre-charge (from surplus, else from the grid).
  * Surplus: store it, then sell, then curtail what the lines can't carry.
  * Deficit: prices are made carbon-aware (price + grid emission factor x carbon price).
    Expensive hour -> discharge; cheap hour -> buy and, if the evening is much dearer, charge;
    in between -> discharge only energy not needed for the upcoming peak.
  * If still beyond the import limit: demand response, then shift load to a later hour with
    headroom, then controlled non-critical cuts (all cheaper than uncontrolled shedding).
"""
from __future__ import annotations

import numpy as np

from ..sim.clock import hhmm
from ..sim.decision import BatteryCommand, Decision, MaintenanceCommand, ShiftCommand
from .base import Agent

NORMAL_RESERVE = 0.15
RISK_LOOKAHEAD = 24          # steps (6 h)
PRICE_WINDOW = 48            # steps (12 h) used to judge cheap vs expensive
CHEAP_Q, EXPENSIVE_Q = 0.35, 0.70
RESERVE_MARGIN = 1.15


class RuleBasedAgent(Agent):
    name = "rule_based"
    label = "Rule-based operator"
    description = ("Heuristic dispatcher: stores surplus and cheap energy, discharges in expensive hours, "
                   "builds battery reserve ahead of forecast shortfalls (P10/P90), uses DR / load shifting "
                   "when the grid is constrained, and dispatches crews. Also the safety fallback.")

    # strategy knobs: a planner (LLM strategist + digital twin) may change these per situation
    reserve_floor: float = NORMAL_RESERVE
    cheap_q: float = CHEAP_Q
    dear_q: float = EXPENSIVE_Q
    risk_lookahead: int = RISK_LOOKAHEAD

    def decide_sync(self, obs: dict) -> Decision:
        cur, fc = obs["current"], obs["forecast"]
        mkt, grid = cur["market"], cur["grid"]
        dt = obs["dt_hours"]
        t = obs["step"]
        d = Decision(agent=self.name, mode="normal")
        why = d.reasons

        bats = cur["batteries"]
        online = {k: b for k, b in bats.items() if b["status"] == "online"}
        lim_imp, lim_exp = grid["import_limit_mw"], grid["export_limit_mw"]
        net = cur["net_position_mw"]

        # carbon-aware effective import price, now and over the next 12 h
        eff_now = mkt["price"] + mkt["grid_ef"] * mkt["carbon_price"]
        H = fc["horizon_steps"]
        eff_fc = (np.array(fc["price"]["p50"]) + np.array(fc["grid_ef"]) * np.array(fc["carbon_price"]))
        ref = np.concatenate(([eff_now], eff_fc[:PRICE_WINDOW]))
        cheap, dear = np.quantile(ref, self.cheap_q), np.quantile(ref, self.dear_q)

        # ---------------------------------------------------------------- maintenance
        self._maintenance(obs, d)

        # ---------------------------------------------------------------- risk look-ahead
        h = min(self.risk_lookahead, H)
        ren_p10 = np.array(fc["solar_total_mw"]["p10"][:h]) + np.array(fc["wind_total_mw"]["p10"][:h])
        dem_p90 = np.array(fc["demand_total_mw"]["p90"][:h])
        gap = np.maximum(0.0, dem_p90 - ren_p10 - np.array(fc["import_limit_mw"][:h]))
        need_mwh = float(gap.sum() * dt)
        usable_cap = sum((b["soc_max"] - b["soc_min"]) * b["effective_energy_mwh"] * b["eta_discharge"]
                         for b in online.values())
        target_mwh = min(need_mwh * RESERVE_MARGIN, usable_cap)
        over_now = max(0.0, -net - lim_imp)   # shortfall beyond the grid right now

        reserve = {}
        cap_total = sum(b["effective_energy_mwh"] for b in online.values()) or 1.0
        for bid, b in online.items():
            share = b["effective_energy_mwh"] / cap_total
            if target_mwh > 1.0:
                r = b["soc_min"] + target_mwh * share / (b["effective_energy_mwh"] * b["eta_discharge"])
                reserve[bid] = float(np.clip(r, self.reserve_floor, b["soc_max"]))
            else:
                reserve[bid] = self.reserve_floor
        if target_mwh > 1.0:
            d.mode = "reliability_prep"
            first = int(np.argmax(gap > 0))
            why.append(f"Risk: up to {need_mwh:.0f} MWh beyond grid limits in next {h * dt:.0f} h "
                       f"(from {fc['times'][first]}); holding {target_mwh:.0f} MWh battery reserve")
        if over_now > 0:
            d.mode = "shortfall"
            why.append(f"Deficit exceeds tie-line limit ({lim_imp:.0f} MW) by {over_now:.0f} MW: using reserve")

        def dis_cap(bid: str, floor: float) -> float:
            b = online[bid]
            return max(0.0, min(b["power_mw"], (b["soc"] - floor) * b["effective_energy_mwh"]
                                * b["eta_discharge"] / dt))

        def precharge_mw(bid: str) -> float:
            b = online[bid]
            need = max(0.0, reserve[bid] - b["soc"]) * b["effective_energy_mwh"] / b["eta_charge"] / dt
            return min(need, b["max_charge_mw"])

        charge = {bid: 0.0 for bid in online}
        discharge = {bid: 0.0 for bid in online}

        # ---------------------------------------------------------------- surplus
        if net >= 0:
            room = {bid: b["max_charge_mw"] for bid, b in online.items()}
            store = min(net, sum(room.values()))
            self._split(store, room, charge)
            rem = net - store
            sell = min(rem, lim_exp)
            curtail = rem - sell
            extra = 0.0
            if d.mode == "reliability_prep":  # surplus alone can't fill the reserve: top up from the grid
                left = {bid: max(0.0, min(precharge_mw(bid), room[bid] - charge[bid])) for bid in online}
                extra = min(sum(left.values()), lim_imp)
                if extra > 0.5:
                    self._split(extra, left, charge)
                    d.buy_mw = extra
                else:
                    extra = 0.0
            d.sell_mw = sell
            if store > 0.5:
                why.append(f"Surplus {net:.0f} MW: storing {store:.0f} MW")
            if sell > 0.5:
                why.append(f"Selling {sell:.0f} MW at ₹{mkt['price']:,.0f}/MWh")
            if extra > 0.5:
                why.append(f"Pre-charging {extra:.0f} MW from the grid for the coming risk window")
            if curtail > 0.5:
                sol, wnd = cur["solar_total_mw"], cur["wind_total_mw"]
                tot = max(sol + wnd, 1e-6)
                d.curtail_mw = {"solar": curtail * sol / tot, "wind": curtail * wnd / tot}
                why.append(f"Export limit {lim_exp:.0f} MW reached and storage full: curtailing {curtail:.0f} MW")

        # ---------------------------------------------------------------- deficit
        else:
            deficit = -net
            # energy above the reserve is free to use; the reserve itself only for a shortfall now
            caps_res = {bid: dis_cap(bid, reserve[bid]) for bid in online}
            caps_min = {bid: dis_cap(bid, b["soc_min"]) for bid, b in online.items()}
            if eff_now >= dear:
                econ = min(deficit, sum(caps_res.values()))
                if econ > 0.5:
                    why.append(f"Expensive hour (₹{eff_now:,.0f}/MWh incl. carbon): discharging {econ:.0f} MW")
            elif eff_now <= cheap:
                econ = 0.0
            else:
                # keep what the upcoming expensive hours need, discharge only the spare
                w = min(PRICE_WINDOW, H)
                ren50 = np.array(fc["solar_total_mw"]["p50"][:w]) + np.array(fc["wind_total_mw"]["p50"][:w])
                dem50 = np.array(fc["demand_total_mw"]["p50"][:w])
                peak_need = float(np.sum(np.maximum(0.0, dem50 - ren50)[eff_fc[:w] >= dear]) * dt)
                stored = sum(max(0.0, online[b]["soc"] - reserve[b]) * online[b]["effective_energy_mwh"]
                             * online[b]["eta_discharge"] for b in online)
                econ = float(np.clip((stored - peak_need) / dt, 0.0, deficit))
                if econ > 0.5:
                    why.append(f"Spare battery energy beyond the evening need: discharging {econ:.0f} MW")
            econ = min(econ, sum(caps_res.values()))
            flex_first = 0.0
            if over_now > 0:  # must cover the part the grid can't carry, dipping into reserve if needed
                stored_all = sum(max(0.0, b["soc"] - b["soc_min"]) * b["effective_energy_mwh"]
                                 * b["eta_discharge"] for b in online.values())
                need_total = need_mwh + over_now * dt
                if need_total > stored_all + 1.0:
                    # batteries alone won't last the whole event: ration them by blending in DR now
                    ration = 1.0 - stored_all / need_total
                    flex_first = self._flex(obs, d, over_now * ration, eff_fc, allow_cuts=False)
                    if flex_first > 0.5:
                        why.append(f"Rationing storage: {stored_all:.0f} MWh stored vs {need_total:.0f} MWh "
                                   f"needed, so {flex_first:.0f} MW comes from flexible demand")
                dis_total = min(max(econ, over_now - flex_first), sum(caps_min.values()))
                self._split(dis_total, caps_min, discharge)
                reserve = {bid: b["soc_min"] for bid, b in online.items()}  # release the floor this step
            else:
                dis_total = econ
                self._split(dis_total, caps_res, discharge)
            buy = deficit - dis_total - flex_first

            # grid charging (never while short)
            gc = 0.0
            if over_now == 0:
                want, label = {}, ""
                if d.mode == "reliability_prep":
                    want = {bid: precharge_mw(bid) for bid in online}
                    label = "Pre-charging {gc:.0f} MW from the grid ahead of the risk window"
                elif eff_now <= cheap and eff_fc[:PRICE_WINDOW].size:
                    best = float(np.max(eff_fc[:PRICE_WINDOW]))
                    rt = 0.94 * 0.94
                    deg = max(b["degradation_cost_per_mwh"] for b in online.values()) if online else 0
                    if best * rt - deg > eff_now * 1.1:
                        want = {bid: (b["max_charge_mw"] if b["soc"] < 0.9 else 0.0) for bid, b in online.items()}
                        label = (f"Cheap hour (₹{eff_now:,.0f}): charging {{gc:.0f}} MW for the "
                                 f"₹{best:,.0f} peak (after losses and wear)")
                if sum(want.values()) > 0.5:
                    # don't discharge and grid-charge at once; then charge only within the tie-line limit
                    for bid in discharge:
                        buy += discharge[bid]
                        discharge[bid] = 0.0
                    gc = min(sum(want.values()), max(0.0, lim_imp - buy))
                    if gc > 0.5:
                        self._split(gc, want, charge)
                        why.append(label.format(gc=gc))
                    else:
                        gc = 0.0
            buy += gc

            # still beyond the tie-line -> demand-side flexibility
            over = max(0.0, buy - lim_imp)
            if over > 0.5:
                covered = self._flex(obs, d, over, eff_fc)
                buy -= covered
            elif eff_now > mkt["dr_incentive"] + 7_000:  # price so high DR beats buying (incl. lost tariff)
                buy -= self._flex(obs, d, min(buy, 30.0), eff_fc, economic=True)
            d.buy_mw = max(0.0, min(buy, lim_imp))
            if d.buy_mw > 0.5 and not any("Buying" in r for r in why):
                why.append(f"Buying {d.buy_mw:.0f} MW at ₹{mkt['price']:,.0f}/MWh")

        # ---------------------------------------------------------------- battery commands
        for bid in bats:
            if bid not in online:
                continue
            if charge.get(bid, 0) > 0.05:
                d.battery[bid] = BatteryCommand("charge", charge[bid], reserve[bid])
            elif discharge.get(bid, 0) > 0.05:
                d.battery[bid] = BatteryCommand("discharge", discharge[bid], reserve[bid])
            else:
                d.battery[bid] = BatteryCommand("idle", 0.0, reserve[bid])
        if d.mode == "normal" and eff_now >= dear:
            d.mode = "peak_saver"
        elif d.mode == "normal" and eff_now <= cheap:
            d.mode = "cheap_hour"
        if not why:
            why.append("Balanced: no action needed")
        return d

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _split(total: float, caps: dict[str, float], out: dict[str, float]) -> None:
        """Split `total` across assets in proportion to their capacity, adding to `out`."""
        cap_sum = sum(max(0.0, c) for c in caps.values())
        if total <= 0 or cap_sum <= 0:
            return
        for k, c in caps.items():
            out[k] = out.get(k, 0.0) + total * max(0.0, c) / cap_sum

    def _flex(self, obs: dict, d: Decision, need: float, eff_fc: np.ndarray, economic: bool = False,
              allow_cuts: bool = True) -> float:
        """Cover `need` MW with DR, then load shifting, then controlled non-critical cuts.
        Additive: safe to call more than once per decision."""
        cur, fc = obs["current"], obs["forecast"]
        demand = cur["demand"]
        covered = 0.0
        shifted_by = lambda cid: sum(s.mw for s in d.shift_load if s.consumer_id == cid)  # noqa: E731
        # 1. demand response (0.98: margin for measurement noise)
        dr_new = 0.0
        for cid, c in sorted(demand.items(), key=lambda kv: -kv[1]["dr_available_mw"]):
            if covered >= need:
                break
            room = c["dr_available_mw"] * 0.98 - d.demand_response_mw.get(cid, 0.0)
            take = min(max(0.0, room), need - covered)
            if take > 0.1:
                d.demand_response_mw[cid] = d.demand_response_mw.get(cid, 0.0) + take
                covered += take
                dr_new += take
        if dr_new > 0.1:
            d.reasons.append(f"Demand response {dr_new:.0f} MW "
                             f"({'price too high' if economic else 'grid limit reached'})")
        if economic or covered >= need:
            return covered
        # 2. shift to the cheapest later step that still has grid headroom
        H = min(fc["horizon_steps"], 32, obs["steps_remaining"] - 1)  # never past the end of the run
        if H <= 0:
            return covered
        ren = np.array(fc["solar_total_mw"]["p50"][:H]) + np.array(fc["wind_total_mw"]["p50"][:H])
        dem = np.array(fc["demand_total_mw"]["p90"][:H])
        headroom = np.array(fc["import_limit_mw"][:H]) - (dem - ren)
        shifted = 0.0
        for idx in np.argsort(eff_fc[:H]):
            if headroom[idx] < 10 or idx < 3:
                continue
            for cid, c in sorted(demand.items(), key=lambda kv: -kv[1]["shiftable_mw"]):
                if covered >= need:
                    break
                room = c["shiftable_mw"] * 0.98 - shifted_by(cid)
                take = min(max(0.0, room), need - covered, headroom[idx] * 0.5 - shifted)
                if take > 0.1:
                    d.shift_load.append(ShiftCommand(cid, take, int(idx) + 1))
                    covered += take
                    shifted += take
            if shifted > 0:
                d.reasons.append(f"Shifting {shifted:.0f} MW of flexible load to {fc['times'][int(idx)]}")
            break
        if covered >= need or not allow_cuts:
            return covered
        # 3. controlled non-critical cuts (cheaper than an uncontrolled blackout)
        cut_total = 0.0
        for cid, c in sorted(demand.items(), key=lambda kv: kv[1]["tariff"]):
            if covered >= need:
                break
            room = (c["noncritical_mw"] * 0.98 - d.demand_response_mw.get(cid, 0.0) - shifted_by(cid)
                    - d.reduce_noncritical_mw.get(cid, 0.0))
            take = min(max(0.0, room), need - covered)
            if take > 0.1:
                d.reduce_noncritical_mw[cid] = d.reduce_noncritical_mw.get(cid, 0.0) + take
                covered += take
                cut_total += take
        if cut_total > 0.1:
            d.reasons.append(f"Controlled cut of {cut_total:.0f} MW non-critical load to avoid a blackout")
        return covered

    def _maintenance(self, obs: dict, d: Decision) -> None:
        t = obs["step"]
        for a in obs["alerts"]:
            if a["type"] == "asset_fault" and a.get("status") == "active" \
                    and a["params"].get("repair_eta_step") is None:
                aid = a["params"]["asset_id"]
                d.maintenance.append(MaintenanceCommand("dispatch_inspection", asset_id=aid))
                d.reasons.append(f"{aid} faulted: dispatching inspection crew")
        fc = obs["forecast"]
        for m in obs["maintenance"]:
            if m.get("status") != "planned":
                continue
            start, end = m["start_step"], m["end_step"]
            if end is None or start - t > 24 or start <= t:
                continue
            asset = m["params"].get("asset_id")
            # outage-free forecast: what the farm *would* produce in each candidate window
            series = fc["wind_potential_by_farm_p50"].get(asset) or fc["solar_potential_by_farm_p50"].get(asset)
            if series is None:
                continue
            dur = end - start
            latest = int(m["params"].get("flexible_until", start))
            f0 = fc["start_step"]

            def lost(s: int) -> float | None:
                i = s - f0
                if i < 0 or i + dur > len(series):
                    return None
                return float(sum(series[i:i + dur])) * obs["dt_hours"]

            now = lost(start)
            if now is None:
                continue
            best_s, best = start, now
            for s in range(max(t + 4, f0), latest + 1):  # give the crew at least 1 h notice
                v = lost(s)
                if v is not None and v < best - 1e-6:
                    best_s, best = s, v
            if best_s != start and now - best > max(5.0, 0.1 * now):
                d.maintenance.append(MaintenanceCommand("schedule", maintenance_id=m["id"], start_step=best_s))
                d.reasons.append(f"Moving {asset} maintenance {hhmm(start)} -> {hhmm(best_s)}: "
                                 f"saves ~{now - best:.0f} MWh of clean generation")
