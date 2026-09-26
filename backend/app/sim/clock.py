"""Conversions between simulation steps and wall-clock time (IST)."""
from __future__ import annotations

from datetime import datetime, timedelta

from .constants import IST, STEP_MINUTES, STEPS_PER_DAY


def at(hour: int, minute: int = 0, day: int = 0) -> int:
    """Step index for a clock time, e.g. at(17, 30) -> 70."""
    return day * STEPS_PER_DAY + (hour * 60 + minute) // STEP_MINUTES


def hhmm(step: int) -> str:
    minutes = (step % STEPS_PER_DAY) * STEP_MINUTES
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def label(step: int) -> str:
    """'D1 17:30' style label (day numbers start at 1)."""
    return f"D{step // STEPS_PER_DAY + 1} {hhmm(step)}"


def to_datetime(start: datetime, step: int) -> datetime:
    return start + timedelta(minutes=STEP_MINUTES * step)


def start_of(date_str: str) -> datetime:
    y, m, d = (int(x) for x in date_str.split("-"))
    return datetime(y, m, d, tzinfo=IST)
