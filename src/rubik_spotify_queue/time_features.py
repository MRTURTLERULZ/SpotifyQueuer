"""Cyclic listening-time features."""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo


def local_time_parts(ts: datetime, timezone_name: str) -> dict[str, float | int]:
    local = ts.astimezone(ZoneInfo(timezone_name))
    hour = int(local.hour)
    dow = int(local.weekday())
    return {
        "hour": hour,
        "day_of_week": dow,
        "hour_sin": float(math.sin(2 * math.pi * hour / 24.0)),
        "hour_cos": float(math.cos(2 * math.pi * hour / 24.0)),
        "day_sin": float(math.sin(2 * math.pi * dow / 7.0)),
        "day_cos": float(math.cos(2 * math.pi * dow / 7.0)),
    }


def within_hour_window(hour: int, start: int, end: int) -> bool:
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end

