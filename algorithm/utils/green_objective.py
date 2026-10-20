from datetime import datetime

from ortools.sat.python import cp_model

from algorithm.utils.tariff import daily_tariff


OBJECTIVE_LIMIT_MAX_TIME_SECONDS = 30 * 60
GREEN_TARIFF_CYCLE_MINUTES = 24 * 60


def get_green_time_value_info(jobs_data, scale_indicator=60):
    """Collect green-objective time-value settings for beta calculation and log reuse."""
    whole_info = jobs_data.get("info", {})
    valley_price = float(whole_info.get("valley_price", 1))
    makespan_60min_cost_in_valley_price_units = float(
        whole_info.get("makespan_60min_cost_in_valley_price_units", 2.0)
    )
    hourly_time_value_yuan = makespan_60min_cost_in_valley_price_units * valley_price
    beta = int(round(hourly_time_value_yuan * 1000 / scale_indicator))
    return {
        "makespan_60min_cost_in_valley_price_units": makespan_60min_cost_in_valley_price_units,
        "valley_price": valley_price,
        "hourly_time_value_yuan": hourly_time_value_yuan,
        "beta": max(0, beta),
    }


def compute_green_schedule_horizon(jobs_data):
    """Compute the finite scheduling horizon shared by time variables and peak/off-peak carbon segments."""
    processing_upper = sum(
        max(int(duration) for duration in durations)
        for job in jobs_data["detail"].values()
        for _, durations, _ in job
    )
    info = jobs_data.get("info", {})
    job_available_time = info.get("job_avail_time", {}) or {}
    machines_available_time = info.get("machines_available_time", []) or []
    release_upper = max(
        [
            0,
            *(int(value) for value in job_available_time.values()),
            *(int(value) for value in machines_available_time),
        ]
    )
    schedule_horizon = release_upper + processing_upper + GREEN_TARIFF_CYCLE_MINUTES
    if schedule_horizon < 0:
        raise ValueError(f"green schedule horizon must be non-negative, got {schedule_horizon}")
    return schedule_horizon, release_upper, processing_upper


def build_price_segments_for_horizon(jobs_data, horizon_upper):
    """Generate price segments over ``[0, horizon_upper)`` using the current peak/off-peak tariff configuration."""
    whole_info = jobs_data.get("info", {})
    if "tariff_periods" in whole_info:
        day = daily_tariff(whole_info)
        dt = datetime.strptime(whole_info.get("cur_time_stamp", "2026-03-24 00:00:00"), "%Y-%m-%d %H:%M:%S")
        origin = dt.hour * 60 + dt.minute
        # Keep only tariff-category changes; do not split continuous off-peak periods at midnight.
        boundaries = [start for start, _, _ in day[1:]]
        if day[0][2] != day[-1][2]:
            boundaries.append(0)
        segments = []
        cur = 0
        while cur < horizon_upper:
            minute = (origin + cur) % 1440
            key = next(key for start, end, key in day if start <= minute < end)
            distance = min(((boundary - minute - 1) % 1440) + 1 for boundary in boundaries) if boundaries else horizon_upper - cur
            end = min(horizon_upper, cur + distance)
            segments.append((cur, end, float(whole_info.get(key, {"peak_price": 2, "valley_price": 1}.get(key)))))
            cur = end
        return segments
    peak_price = whole_info.get("peak_price", 2)
    valley_price = whole_info.get("valley_price", 1)

    valley_start = whole_info.get("valley_start", 22)
    valley_end = whole_info.get("valley_end", 8)
    minute_boundaries = (
        "valley_start_minute" in whole_info
        or "valley_end_minute" in whole_info
        or any(isinstance(value, str) and ":" in value for value in (valley_start, valley_end))
    )
    if minute_boundaries:
        valley_start_min = _get_tariff_boundary_minutes(whole_info, "valley_start")
        valley_end_min = _get_tariff_boundary_minutes(whole_info, "valley_end")
    else:
        # Ordinary mode preserves integer-hour tariff settings and peak/off-peak classification.
        valley_start_min = valley_start * 60
        valley_end_min = valley_end * 60
    dt = datetime.strptime(whole_info.get("cur_time_stamp", "2026-03-24 00:00:00"), "%Y-%m-%d %H:%M:%S")
    origin_min = dt.hour * 60 + dt.minute

    def is_valley(min_of_day: int) -> bool:
        if minute_boundaries and valley_start_min < valley_end_min:
            return valley_start_min <= min_of_day < valley_end_min
        return min_of_day >= valley_start_min or min_of_day < valley_end_min

    def next_switch_offset(abs_min: int) -> int:
        m = (origin_min + abs_min) % 1440
        target = valley_end_min if is_valley(m) else valley_start_min
        if m < target:
            return target - m
        return 1440 - m + target

    price_segments = []
    cur = 0
    while cur < horizon_upper:
        m = (origin_min + cur) % 1440
        price = valley_price if is_valley(m) else peak_price
        delta_tou = min(horizon_upper - cur, next_switch_offset(cur))
        price_segments.append((cur, cur + delta_tou, price))
        cur += delta_tou
    return price_segments


def _get_tariff_boundary_minutes(whole_info, boundary_name):

    minute_key = f"{boundary_name}_minute"
    if minute_key in whole_info:
        value = int(whole_info[minute_key])
    else:
        raw_value = whole_info.get(boundary_name, 22 if boundary_name == "valley_start" else 8)
        if isinstance(raw_value, str) and ":" in raw_value:
            parts = raw_value.split(":")
            if len(parts) != 2:
                raise ValueError(f"invalid tariff boundary {boundary_name}: {raw_value!r}")
            hour, minute = (int(part) for part in parts)
            if not 0 <= hour < 24 or not 0 <= minute < 60:
                raise ValueError(f"invalid tariff boundary {boundary_name}: {raw_value!r}")
            value = hour * 60 + minute
        else:
            value = int(round(float(raw_value) * 60))

    if not 0 <= value < GREEN_TARIFF_CYCLE_MINUTES:
        raise ValueError(f"invalid tariff boundary {boundary_name}: {value} minutes")
    return value


def shift_valley_window_minutes(anchor_time_stamp, valley_start, valley_end):
    """Shift the off-peak window under virtual 12:00 to the anchor wall-clock time; the result is in minutes within the day."""
    anchor_dt = datetime.strptime(anchor_time_stamp, "%Y-%m-%d %H:%M:%S")
    anchor_minute_of_day = anchor_dt.hour * 60 + anchor_dt.minute
    virtual_origin_minute = 12 * 60
    valley_start_min = _get_tariff_boundary_minutes({"valley_start": valley_start}, "valley_start")
    valley_end_min = _get_tariff_boundary_minutes({"valley_end": valley_end}, "valley_end")
    return (
        (anchor_minute_of_day + valley_start_min - virtual_origin_minute) % GREEN_TARIFF_CYCLE_MINUTES,
        (anchor_minute_of_day + valley_end_min - virtual_origin_minute) % GREEN_TARIFF_CYCLE_MINUTES,
    )


def add_green_objective_terms(model, jobs_data, machine_batches, horizon, scale_indicator=60):
    """Add peak/off-peak tariff overlap linearization and carbon variables to an existing base model."""
    machine_power_info = {m_id: p for m_id, p in enumerate(jobs_data["info"]["machine_power_info"])}
    price_segments = build_price_segments_for_horizon(jobs_data, horizon)
    carbon_terms = []
    green_overlap_hints = []

    for m_id, batch_list in machine_batches.items():
        power_int = int(round(machine_power_info.get(m_id, 1000.0)))
        for batch_idx, batch in enumerate(batch_list):
            for seg_start, seg_end, price in price_segments:
                seg_len = seg_end - seg_start
                min_end = model.NewIntVar(0, horizon, f"minE_{batch.interval.Name()}_{seg_start}")
                model.AddMinEquality(min_end, [batch.end, seg_end])
                max_start = model.NewIntVar(0, horizon, f"maxS_{batch.interval.Name()}_{seg_start}")
                model.AddMaxEquality(max_start, [batch.start, seg_start])
                diff = model.NewIntVar(-horizon, horizon, f"diff_{batch.interval.Name()}_{seg_start}")
                model.Add(diff == min_end - max_start)
                overlap = model.NewIntVar(0, seg_len, f"ov_{batch.interval.Name()}_{seg_start}")
                model.AddMaxEquality(overlap, [diff, 0])
                model.Add(overlap == 0).OnlyEnforceIf(batch.presence.Not())
                carbon_coefficient = int(round(price * power_int / scale_indicator))
                carbon_terms.append(overlap * carbon_coefficient)
                green_overlap_hints.append(
                    {
                        "machine_id": m_id,
                        "batch_idx": batch_idx,
                        "seg_start": seg_start,
                        "seg_end": seg_end,
                        "min_end": min_end,
                        "max_start": max_start,
                        "diff": diff,
                        "overlap": overlap,
                        "carbon_coefficient": carbon_coefficient,
                    }
                )

    carbon_emission = model.NewIntVar(0, 10**18, "carbon_emission")
    model.Add(carbon_emission == sum(carbon_terms))
    return carbon_emission, green_overlap_hints



_get_green_time_value_info = get_green_time_value_info
_build_price_segments_for_horizon = build_price_segments_for_horizon
