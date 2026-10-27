"""FJSP-B scheduler with a working calendar and deterministic ECT rule for human-like scheduling."""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Iterable

import torch


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

from rl.env.case_generator import CaseGenerator
from rl.utils.draw_gantt import transfer_data_form

from rl.env.fjsp_env import FJSPEnv


HUMAN_DEFAULT_WORKDAYS = (0, 1, 2, 3, 4)
HUMAN_DEFAULT_WORK_START_MINUTE = 9 * 60
HUMAN_DEFAULT_WORK_END_MINUTE = 17 * 60
HUMAN_TIME_EPSILON = 1e-6


class HumanFJSPEnv(FJSPEnv):
    """Add an absolute-time working calendar for human-like scheduling without changing the generic FJSPEnv."""

    def __init__(
        self,
        case,
        env_params,
        data_source="case",
        train=False,
        *,
        absolute_start_time: datetime,
        workdays: Iterable[int] = (0, 1, 2, 3, 4),
        work_start_minute: int = 9 * 60,
        work_end_minute: int = 17 * 60,
    ):
        # Save Human calendar settings first; the base constructor invokes time advancement,
        # so calendar-wait tensors remain None until base batch initialization is complete.
        self.human_start_time = self._coerce_datetime(absolute_start_time)
        self.human_workdays = frozenset(int(day) for day in workdays)
        self.human_work_start_minute = int(work_start_minute)
        self.human_work_end_minute = int(work_end_minute)
        if not self.human_workdays.issubset(set(range(7))):
            raise ValueError("workdays must contain only datetime.weekday() values in [0, 6]")
        if not 0 <= self.human_work_start_minute < 24 * 60:
            raise ValueError("work_start_minute must be in [0, 1440)")
        if not 0 < self.human_work_end_minute <= 24 * 60:
            raise ValueError("work_end_minute must be in (0, 1440]")
        if self.human_work_start_minute >= self.human_work_end_minute:
            raise ValueError("work_start_minute must be earlier than work_end_minute")

        self._human_calendar_wait_until = None
        super().__init__(case, env_params, data_source=data_source, train=train)

        # Calendar events use one relative minute value per batch; inf means no calendar wait requested.
        self._human_calendar_wait_until = torch.full(
            (self.batch_size,), float("inf"), dtype=self.time.dtype
        )

    @staticmethod
    def _coerce_datetime(value: datetime | str) -> datetime:
        """Use one absolute simulation start for Human to avoid implicit dependence on the current system time."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError as exc:
                raise ValueError(
                    "absolute_start_time must use '%Y-%m-%d %H:%M:%S'"
                ) from exc
        raise TypeError("absolute_start_time must be a datetime or formatted string")

    def relative_minute_to_datetime(self, relative_minute: float, batch_id: int = 0) -> datetime:
        """Convert environment-relative minutes to an absolute simulation time."""
        del batch_id  # Human currently uses one batch; keep the batch interface for behavioral tests.
        return self.human_start_time + timedelta(minutes=float(relative_minute))

    def is_human_work_time(self, relative_minute: float | datetime, batch_id: int = 0) -> bool:
        """Check whether a human may start an operation at the specified relative minute / absolute time."""
        if isinstance(relative_minute, datetime):
            current = relative_minute
        else:
            current = self.relative_minute_to_datetime(float(relative_minute), batch_id)
        minute_of_day = (
            current.hour * 60
            + current.minute
            + current.second / 60.0
            + current.microsecond / 60_000_000.0
        )
        return current.weekday() in self.human_workdays and (
            self.human_work_start_minute <= minute_of_day < self.human_work_end_minute
        )

    def next_human_work_start(
        self, relative_minute: float | None = None, batch_id: int = 0
    ) -> float:
        """Return the relative minute for the next workday at 09:00, strictly after the current time."""
        if relative_minute is None:
            relative_minute = float(self.time[batch_id].item())
        current = self.relative_minute_to_datetime(float(relative_minute), batch_id)
        candidate = current.replace(
            hour=self.human_work_start_minute // 60,
            minute=self.human_work_start_minute % 60,
            second=0,
            microsecond=0,
        )

        # Before work starts on a workday, advance to 09:00; otherwise search from the next day.
        if current.weekday() not in self.human_workdays or current >= candidate:
            candidate += timedelta(days=1)
        while candidate.weekday() not in self.human_workdays:
            candidate += timedelta(days=1)

        result = (candidate - self.human_start_time).total_seconds() / 60.0
        if result <= float(relative_minute) + 1e-6:
            # Floating-point guard: calendar events must strictly advance time to prevent loops.
            candidate += timedelta(days=1)
            while candidate.weekday() not in self.human_workdays:
                candidate += timedelta(days=1)
            result = (candidate - self.human_start_time).total_seconds() / 60.0
        return result

    @property
    def human_calendar_wait_until(self) -> torch.Tensor:
        """Expose the current Human calendar wait target as read-only for diagnostics and behavior tests."""
        if self._human_calendar_wait_until is None:
            return torch.empty(0)
        return self._human_calendar_wait_until

    def request_human_calendar_wait(self, target_relative_minute: float, batch_id: int = 0) -> None:
        """Register one next-workday event for shared queries by base-class drop/transfer logic."""
        if self._human_calendar_wait_until is None:
            raise RuntimeError("Human calendar is not initialized")
        current = float(self.time[batch_id].item())
        target = float(target_relative_minute)
        if not math.isfinite(target) or target <= current + 1e-6:
            raise ValueError(
                f"human calendar target must be finite and later than current time: "
                f"current={current}, target={target}"
            )
        self._human_calendar_wait_until[batch_id] = target

    def clear_human_calendar_wait(self, batch_id: int | None = None) -> None:
        """Clear a consumed Human calendar wait to avoid triggering the same event again."""
        if self._human_calendar_wait_until is None:
            return
        if batch_id is None:
            self._human_calendar_wait_until.fill_(float("inf"))
        else:
            self._human_calendar_wait_until[batch_id] = float("inf")

    def _get_next_job_available_time(self):
        """Merge an explicitly requested Human workday event into the real base-class events."""
        next_job_time = super()._get_next_job_available_time()
        if self._human_calendar_wait_until is None:
            return next_job_time

        reached = self._human_calendar_wait_until <= self.time + 1e-6
        if reached.any():
            self._human_calendar_wait_until[reached] = float("inf")
        return torch.minimum(next_job_time, self._human_calendar_wait_until)

    def _get_drop_boundary_adjust_mask(
        self,
        flag_need_trans,
        earliest_available_time,
        fallback_time,
        earliest_real_event_time,
    ):

        del earliest_available_time, fallback_time
        return flag_need_trans & (~torch.isfinite(earliest_real_event_time))

    def chain_calendar_start_times(
        self,
        chain_ops: Iterable[int],
        machine_path: Iterable[int],
        start_relative_minute: float,
        batch_id: int = 0,
    ) -> tuple[list[float], float | None]:
        """Preview the start time of a no-wait chain and return the estimated completion time of its last operation."""
        operations = [int(op) for op in chain_ops]
        machines = [int(machine) for machine in machine_path]
        if len(operations) != len(machines):
            raise ValueError("chain_ops and machine_path must have equal lengths")

        now = float(self.time[batch_id].item())
        current = float(start_relative_minute)
        starts: list[float] = []
        for operation, machine in zip(operations, machines):
            starts.append(current)
            if not self.is_human_work_time(current, batch_id):
                return starts, None
            duration = float(self.proc_times_batch[batch_id, operation, machine].item())
            if duration <= 0:
                return starts, None

            machine_state = self.partial_machines_batch[batch_id, machine]
            joins_active_batch = (
                current <= now + 1e-6
                and machine_state[4].item() == 1.0
                and current <= machine_state[6].item() + 1e-6
            )
            if joins_active_batch:
                current = float(machine_state[7].item())
            else:
                current += duration
        return starts, current

    def reset(self):
        """Clear Human calendar events during reset before asking the base class to restore scheduling state."""
        self.clear_human_calendar_wait()
        return super().reset()

def _parse_human_origin(jobs_data: dict[str, Any]) -> datetime:
    """Strictly parse the absolute simulation start required for dynamic rescheduling."""
    try:
        value = jobs_data["info"]["cur_time_stamp"]
    except (KeyError, TypeError) as exc:
        raise ValueError("jobs_data['info']['cur_time_stamp'] is required for human scheduling") from exc
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            "jobs_data['info']['cur_time_stamp'] must use '%Y-%m-%d %H:%M:%S'"
        ) from exc


def _machine_finish_time(
    env: HumanFJSPEnv, batch_id: int, machine_id: int, start_time: float, duration: float
) -> float:
    """Estimate the current operation’s completion time from the end time of its active batch."""
    machine_state = env.partial_machines_batch[batch_id, machine_id]
    if (
        machine_state[4].item() == 1.0
        and start_time <= float(env.time[batch_id].item()) + HUMAN_TIME_EPSILON
        and start_time <= machine_state[6].item() + HUMAN_TIME_EPSILON
    ):
        return float(machine_state[7].item())
    return start_time + duration


def _candidate_actions(env: HumanFJSPEnv) -> list[dict[str, Any]]:
    """Build Human feasible actions from base-class eligibility and the full chain path."""
    candidates: list[dict[str, Any]] = []
    state = env.state
    for batch_tensor in state.batch_idxes:
        batch_id = int(batch_tensor.item())
        if bool(env.done_batch[batch_id].item()):
            continue
        current_time = float(env.time[batch_id].item())
        if not env.is_human_work_time(current_time, batch_id):
            continue

        eligible = state.eligible_om_pairs_batch[batch_id]
        for job_id in range(env.num_jobs):
            if bool(env.mask_job_procing_batch[batch_id, job_id].item()) or bool(
                env.mask_job_finish_batch[batch_id, job_id].item()
            ):
                continue
            operation = int(env.step_oper_id_of_each_job_batch[batch_id, job_id].item())
            if operation > int(env.end_oper_id_of_each_job_batch[batch_id, job_id].item()):
                continue

            eligible_machines = torch.nonzero(eligible[job_id], as_tuple=False).flatten().tolist()
            is_chain_start = operation in env.chain_first_ops.get(batch_id, set())
            if is_chain_start:
                chain_ops = env.chain_info[batch_id][operation]
                paths_by_machine = (
                    state.chain_feasible_paths_batch.get(batch_id, {})
                    .get(operation, {})
                )
                for machine_id in eligible_machines:
                    all_paths = paths_by_machine.get(int(machine_id), [])
                    for path in all_paths:
                        full_path = (int(machine_id),) + tuple(int(item) for item in path)
                        _, chain_end = env.chain_calendar_start_times(
                            chain_ops,
                            full_path,
                            current_time,
                            batch_id,
                        )
                        if chain_end is None:
                            continue
                        assignments = {
                            int(chain_op): int(chain_machine)
                            for chain_op, chain_machine in zip(chain_ops, full_path)
                        }
                        candidates.append(
                            {
                                # After ECT, sort stably by job, operation, first machine, and complete path.
                                "sort_key": (
                                    float(chain_end),
                                    int(job_id),
                                    int(operation),
                                    int(machine_id),
                                    tuple(full_path),
                                ),
                                "action": (int(operation), int(machine_id), int(job_id), 0),
                                "chain_assignments": {batch_id: assignments},
                            }
                        )
                continue

            for machine_id in eligible_machines:
                machine_id = int(machine_id)
                duration = float(env.proc_times_batch[batch_id, operation, machine_id].item())
                if duration <= 0:
                    continue
                finish_time = _machine_finish_time(
                    env, batch_id, machine_id, current_time, duration
                )
                candidates.append(
                    {
                        # Use estimated completion time for ordinary operations, breaking ties by job, operation, and machine.
                        "sort_key": (
                            float(finish_time),
                            int(job_id),
                            int(operation),
                            int(machine_id),
                            (),
                        ),
                        "action": (int(operation), int(machine_id), int(job_id), 0),
                        "chain_assignments": {},
                    }
                )
    return candidates


def _validate_human_schedule(env: HumanFJSPEnv, validate_result: tuple[Any, ...]) -> None:
    """Add Human working-hours and no-wait-chain assertions after base-class validation."""
    if not validate_result[0]:
        failures = getattr(env, "gantt_validation_failures", [])
        raise RuntimeError(f"human schedule failed env.validate_gantt(): {failures}")

    scheduled_result = validate_result[1][0]
    num_operations = int(env.num_opers_batch[0].item())
    for operation in range(num_operations):
        if scheduled_result[operation, 0].item() != 1:
            raise RuntimeError(f"human schedule left operation {operation} unscheduled")
        start_time = float(scheduled_result[operation, 2].item())
        if not env.is_human_work_time(start_time, 0):
            absolute_time = env.relative_minute_to_datetime(start_time)
            raise RuntimeError(
                f"human operation {operation} starts outside work time: "
                f"relative={start_time}, absolute={absolute_time:%Y-%m-%d %H:%M:%S}"
            )

    # Explicitly check no-wait adjacency; the base validator does not repeat this check.
    for chain_ops in env.chain_info.get(0, {}).values():
        for previous, following in zip(chain_ops, chain_ops[1:]):
            previous_end = float(scheduled_result[previous, 3].item())
            following_start = float(scheduled_result[following, 2].item())
            if abs(previous_end - following_start) > HUMAN_TIME_EPSILON:
                raise RuntimeError(
                    f"human no-wait chain violation: op {previous} ends at {previous_end}, "
                    f"op {following} starts at {following_start}"
                )


def _machines_and_capacities(jobs_data: dict[str, Any]) -> tuple[list[int], dict[int, int]]:
    """Create a zero-based machine list and capacity mapping."""
    raw_capacities = jobs_data["info"]["machine_capacities"]
    if isinstance(raw_capacities, dict):
        batch_capacities = {int(machine) - 1: int(capacity) for machine, capacity in raw_capacities.items()}
    else:
        batch_capacities = {index: int(capacity) for index, capacity in enumerate(raw_capacities)}
    machines_list = sorted(
        {
            int(machine) - 1
            for job in jobs_data["detail"].values()
            for machines, _, *_ in job
            for machine in machines
        }
    )
    return machines_list, batch_capacities


def fespb_human(
    jobs_data: dict[str, Any],
    env_params: dict[str, Any],
    *,
    workdays: Iterable[int] = HUMAN_DEFAULT_WORKDAYS,
    work_start_minute: int = HUMAN_DEFAULT_WORK_START_MINUTE,
    work_end_minute: int = HUMAN_DEFAULT_WORK_END_MINUTE,
):
    """Run deterministic ECT Human scheduling in one environment and return standard FJSP results."""
    start_wall_time = time.time()
    origin = _parse_human_origin(jobs_data)
    num_jobs, num_machines, _ = CaseGenerator.fetch_basic_info(jobs_data)


    human_env_params = copy.deepcopy(env_params)
    human_env_params["batch_size"] = 1
    human_env_params["num_jobs"] = num_jobs
    human_env_params["num_machines"] = num_machines
    env = HumanFJSPEnv(
        case=[jobs_data],
        env_params=human_env_params,
        data_source="dict",
        train=False,
        absolute_start_time=origin,
        workdays=workdays,
        work_start_minute=work_start_minute,
        work_end_minute=work_end_minute,
    )

    max_steps = max(1000, int(env.num_opers_batch[0].item()) * 100)
    iteration = 0
    while not bool(env.done_batch.all().item()):
        iteration += 1
        if iteration > max_steps:
            absolute_time = env.relative_minute_to_datetime(float(env.time[0].item()))
            raise RuntimeError(
                f"human scheduling exceeded progress limit at "
                f"{absolute_time:%Y-%m-%d %H:%M:%S}; iteration={iteration}"
            )

        before_time = float(env.time[0].item())
        before_scheduled = int(env.num_scheduled_oper_batch[0].item())
        candidates = _candidate_actions(env)
        if candidates:
            selected = min(candidates, key=lambda item: item["sort_key"])
            # FJSPEnv.step() expects basic_actions with shape [4, batch_size].
            basic_action = torch.tensor(selected["action"], dtype=torch.long).reshape(4, 1)
            chain_assignments = selected["chain_assignments"]
        else:
            # When no human-startable action exists, wait for the next workday and reuse base drop-event advancement.
            target_time = env.next_human_work_start(before_time, 0)
            env.request_human_calendar_wait(target_time, 0)
            basic_action = torch.tensor((0, 0, 0, 1), dtype=torch.long).reshape(4, 1)
            chain_assignments = {}

        env.step((basic_action, chain_assignments))
        after_time = float(env.time[0].item())
        after_scheduled = int(env.num_scheduled_oper_batch[0].item())
        if (
            abs(after_time - before_time) <= HUMAN_TIME_EPSILON
            and after_scheduled == before_scheduled
            and not bool(env.done_batch[0].item())
        ):
            absolute_time = env.relative_minute_to_datetime(after_time)
            pending_jobs = int((~env.mask_job_finish_batch[0]).sum().item())
            wait_until = float(env.human_calendar_wait_until[0].item())
            raise RuntimeError(
                "human scheduling made no progress: "
                f"absolute_time={absolute_time:%Y-%m-%d %H:%M:%S}, "
                f"pending_jobs={pending_jobs}, calendar_wait_until={wait_until}"
            )

    validate_result = env.validate_gantt()
    _validate_human_schedule(env, validate_result)
    scheduled_result = validate_result[1][0].cpu()
    first_oper_ids = validate_result[2][0].cpu()
    capacities = validate_result[3][0].cpu()
    fjsp_result = (
        human_env_params["num_jobs"],
        human_env_params["num_machines"],
        scheduled_result,
        first_oper_ids,
        capacities,
    )
    ws_table, makespan = transfer_data_form(
        fjsp_result[0], fjsp_result[1], fjsp_result[2], fjsp_result[3]
    )
    machines_list, batch_capacities = _machines_and_capacities(jobs_data)
    print(f"Finish human scheduling, spend_time: {time.time() - start_wall_time:.3f}s")
    return makespan, ws_table, batch_capacities, machines_list, fjsp_result
