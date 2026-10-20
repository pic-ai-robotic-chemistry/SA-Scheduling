"""Minute-resolution daily repeating time-of-use tariffs; solver-independent."""
import copy
import math
from pathlib import Path

import json5


TARIFF_FIELDS = ("peak_price", "valley_price", "flat_price", "valley_start", "valley_end",
                 "valley_start_minute", "valley_end_minute", "tariff_periods")


def clock_minute(value, *, allow_24=False):
    if isinstance(value, str) and ":" in value:
        hour, minute = map(int, value.split(":"))
        if not 0 <= minute < 60 or not 0 <= hour <= 24 or (hour == 24 and minute):
            raise ValueError(f"Invalid period boundary: {value!r}")
        result = hour * 60 + minute
    else:
        number = float(value) * 60
        if not math.isfinite(number) or number != int(number):
            raise ValueError(f"Period boundaries must have minute precision: {value!r}")
        result = int(number)
    if not 0 <= result < 1440 + int(allow_24):
        raise ValueError(f"Invalid period boundary: {value!r}")
    return result


def daily_tariff(info):
    """Return complete daily segments as (start, end, price_key); validate coverage, overlaps, and prices."""
    periods = info.get("tariff_periods")
    if periods is None:
        start = int(info["valley_start_minute"]) if "valley_start_minute" in info else clock_minute(info.get("valley_start", 22))
        end = int(info["valley_end_minute"]) if "valley_end_minute" in info else clock_minute(info.get("valley_end", 8))
        periods = ([{"start": "00:00", "end": "24:00", "price_key": "valley_price"}]
                   if start == end else
                   [{"start": f"{start // 60:02d}:{start % 60:02d}", "end": f"{end // 60:02d}:{end % 60:02d}", "price_key": "valley_price"},
                    {"start": f"{end // 60:02d}:{end % 60:02d}", "end": f"{start // 60:02d}:{start % 60:02d}", "price_key": "peak_price"}])
    if not isinstance(periods, list) or not periods:
        raise ValueError("tariff_periods must be a nonempty list")
    segments = []
    for period in periods:
        start = clock_minute(period["start"])
        end = clock_minute(period["end"], allow_24=True)
        key = period["price_key"]
        if key not in ("peak_price", "valley_price", "flat_price"):
            raise ValueError(f"Invalid price_key: {key!r}")
        price = float(info.get(key, {"peak_price": 2, "valley_price": 1}.get(key, float("nan"))))
        if not math.isfinite(price) or price < 0:
            raise ValueError(f"{key} must be finite and nonnegative; shoulder periods require an explicit flat_price")
        if start == end:
            raise ValueError("Use 00:00–24:00 for a full-day period; start and end must differ")
        if start < end:
            segments.append((start, end, key))
        else:
            segments.append((start, 1440, key))
            if end:
                segments.append((0, end, key))
    segments.sort()
    cursor = 0
    merged = []
    for start, end, key in segments:
        if start != cursor:
            raise ValueError("Tariff periods must cover the full day without overlaps or gaps")
        if merged and merged[-1][2] == key:
            merged[-1] = (merged[-1][0], end, key)
        else:
            merged.append((start, end, key))
        cursor = end
    if cursor != 1440:
        raise ValueError("Tariff periods must cover the full day")
    return merged


def tariff_snapshot(info):
    return copy.deepcopy({key: info[key] for key in TARIFF_FIELDS if key in info})


def tariff_identity(info):

    day = daily_tariff(info or {})
    return (day, float(info["flat_price"]) if any(key == "flat_price" for _, _, key in day) else None)


def load_tariff_config(power_info, config_dir):
    result = copy.deepcopy(power_info)
    if result.get("tariff_file"):
        path = Path(config_dir) / result["tariff_file"]
        with path.open(encoding="utf-8") as file:
            tariff = json5.load(file)
        result.update({key: tariff[key] for key in TARIFF_FIELDS if key in tariff})
    daily_tariff(result)
    return result


def shift_tariff_periods(info, anchor_time_stamp):

    if "tariff_periods" not in info:
        return
    from datetime import datetime
    anchor = datetime.strptime(anchor_time_stamp, "%Y-%m-%d %H:%M:%S")
    offset = anchor.hour * 60 + anchor.minute - 720
    shifted = []
    for start, end, key in daily_tariff(info):
        if end - start == 1440:
            shifted.append({"start": "00:00", "end": "24:00", "price_key": key})
        else:
            def clock(value):
                value %= 1440
                return f"{value // 60:02d}:{value % 60:02d}"
            shifted.append({"start": clock(start + offset), "end": clock(end + offset), "price_key": key})
    info["tariff_periods"] = shifted
