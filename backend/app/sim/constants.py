"""Time, unit and market constants shared by the whole simulator.

Units used everywhere:
    power  -> MW
    energy -> MWh
    money  -> INR (₹)
    price  -> ₹/MWh   (₹10,000/MWh == ₹10/kWh)
    carbon -> tCO2
"""
from datetime import timedelta, timezone

STEP_MINUTES = 15
DT_H = STEP_MINUTES / 60            # 0.25 h per step
STEPS_PER_HOUR = 60 // STEP_MINUTES  # 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR  # 96
FORECAST_HORIZON = STEPS_PER_DAY     # 24 h look-ahead

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# --- Market & settlement (India-style, simplified; all configurable per scenario) ---
PRICE_CAP = 10_000.0                 # ₹/MWh exchange price cap
PRICE_FLOOR = 500.0
SELL_FACTOR = 0.97                   # we receive 97% of the price when selling (fees, transmission)
DSM_DEFICIT_MULT = 1.5               # under-injection (unscheduled import) charged at 1.5x price...
DSM_DEFICIT_MIN_ADDER = 1_000.0      # ...or price + ₹1000, whichever is higher
DSM_LOW_FREQ_MULT = 2.0              # doubled when grid frequency < LOW_FREQ_HZ
DSM_SURPLUS_MULT = 0.5               # over-injection (unscheduled export) paid at only 50% of price
NOMINAL_FREQ_HZ = 50.0
LOW_FREQ_HZ = 49.90
HIGH_FREQ_HZ = 50.05

# --- Flexibility & reliability costs ---
SHIFT_FEE = 300.0                    # ₹/MWh paid to a consumer for moving load in time
NONCRITICAL_CUT_COST = 12_000.0      # ₹/MWh penalty for a controlled cut of non-critical load
VOLL_NONCRITICAL = 25_000.0          # ₹/MWh value of lost load, uncontrolled shedding (non-critical)
VOLL_CRITICAL = 100_000.0            # ₹/MWh value of lost load, critical load shed
INSPECTION_COST = 40_000.0           # ₹ per crew dispatch
INSPECTION_REPAIR_STEPS = 4          # crew fixes a fault in 1 h
AUTO_REPAIR_STEPS = 32               # without a crew, a fault clears only after 8 h
MAX_SHIFT_STEPS = 32                 # load can be moved at most 8 h later
CURTAILMENT_SHADOW_PRICE = 2_000.0   # ₹/MWh value of wasted clean energy (used only if weighted)

EPS = 1e-6
