"""Cyclic listening-time features."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo


class ContinuousTimeParts(TypedDict):
    hour: int
    day_of_week: int
    hour_float: float
    day_float: float
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float


def continuous_time_parts(ts: datetime, timezone_name: str) -> ContinuousTimeParts:
    local = ts.astimezone(ZoneInfo(timezone_name))
    hour = local.hour
    dow = local.weekday()
    hour_float = (
        float(local.hour)
        + float(local.minute) / 60.0
        + float(local.second) / 3600.0
        + float(local.microsecond) / 3_600_000_000.0
    )
    day_float = float(dow) + hour_float / 24.0
    return {
        "hour": hour,
        "day_of_week": dow,
        "hour_float": hour_float,
        "day_float": day_float,
        "hour_sin": float(math.sin(2 * math.pi * hour_float / 24.0)),
        "hour_cos": float(math.cos(2 * math.pi * hour_float / 24.0)),
        "day_sin": float(math.sin(2 * math.pi * day_float / 7.0)),
        "day_cos": float(math.cos(2 * math.pi * day_float / 7.0)),
    }


def local_time_parts(ts: datetime, timezone_name: str) -> ContinuousTimeParts:
    return continuous_time_parts(ts, timezone_name)


def within_hour_window(hour: int, start: int, end: int) -> bool:
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end
