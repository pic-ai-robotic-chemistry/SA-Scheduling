import copy
from itertools import chain
from collections import defaultdict
from collections.abc import Iterable
from pprint import pprint, pformat
from typing import DefaultDict

from scheduler.scheduler import SchedulerDatabase
from utils.async_log import AsyncLog

from obj.step import Step, StepFactory
from obj.safe_attr import SafeAttr


class Task:
    FIELDS = ["task_group", "expr_id", "expr_no", "expr_name", "status"]

    def __init__(self, logger: AsyncLog, task_data):
        self.logger = logger

        self._assign_fields(task_data)
        self.steps: list[Step] = self.init_steps(task_data.get("steps"))

        self.next_step_idx = 0  # Index of the next step to execute.

        self.step_earlier_completed = False
        self.step_later_completed = False
        self.the_earlier_completed_step = None
        self.the_later_completed_step = None

    def _assign_fields(self, data):
        for field in self.FIELDS:
            setattr(self, field, data.get(field))

    def init_steps(self, step_list) -> list[Step]:
        return [StepFactory.create_step(data) for data in step_list]

    def update(self, task_data):
        self._assign_fields(task_data)

        # Update all steps and next_step_index.
        self.logger.debug(f"Updating step information for task {self.expr_no}.")
        self.next_step_idx = None  # Reset on each update.
        for step, step_data in zip(self.steps, task_data.get("steps", [])):
            step.update(step_data)
            if self.next_step_idx is None and step.get_attr("status").get() in [
                "in_progress",
                "not_start",
                "error",
            ]:
                self.next_step_idx = step.get_attr("fjspb_index").get()
            self.logger.debug(f"Step status: {step}")

        self.step_earlier_completed = False
        self.step_later_completed = False
        self.the_earlier_completed_step = None
        self.the_later_completed_step = None

        if self.next_step_idx is not None:
            if (
                self.steps[self.next_step_idx].get_attr("status").get() in ["not_start"]
                and self.next_step_idx > 0
            ):
                self.step_earlier_completed = (
                    self.steps[self.next_step_idx - 1].get_attr("earlier_completed").get()
                )
                self.step_later_completed = (
                    self.steps[self.next_step_idx - 1].get_attr("later_completed").get()
                )

                if self.step_earlier_completed:
                    self.the_earlier_completed_step = self.steps[self.next_step_idx - 1]
                if self.step_later_completed:
                    self.the_later_completed_step = self.steps[self.next_step_idx - 1]
            elif self.steps[self.next_step_idx].get_attr("status").get() in ["in_progress", "error"]:
                self.step_earlier_completed = (
                    self.steps[self.next_step_idx].get_attr("earlier_completed").get()
                )
                self.step_later_completed = self.steps[self.next_step_idx].get_attr("later_completed").get()

                if self.step_earlier_completed:
                    self.the_earlier_completed_step = self.steps[self.next_step_idx]
                if self.step_later_completed:
                    self.the_later_completed_step = self.steps[self.next_step_idx]

    def get_next_step(self) -> Step | None:
        if self.next_step_idx is not None:
            return self.steps[self.next_step_idx]
        return None

    def get_attr(self, attr_name):
        return SafeAttr(getattr(self, attr_name, None))

    def __str__(self) -> str:
        return (
            "Task(\n"
            + pformat(
                {
                    "task_group": self.task_group,
                    "expr_id": self.expr_id,
                    "expr_no": self.expr_no,
                    "expr_name": self.expr_name,
                    "status": self.status,
                    "steps": self.steps,
                },
                indent=4,
            )
            + "\n)"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TaskFactory:
    @staticmethod
    def create_task(logger: AsyncLog, db: SchedulerDatabase, expr_no: str) -> Task:
        return Task(logger, db.fetch_task_info(expr_no))

    @staticmethod
    def update_task(db: SchedulerDatabase, expr_no: str) -> Task:
        return db.fetch_task_info(expr_no)


class AllTasks:
    def __init__(self, logger: AsyncLog, db: SchedulerDatabase):
        self.logger = logger
        self.db = db

        self.active_tasks: defaultdict[str, list[Task]] = defaultdict(list)

        self.old_task_names = defaultdict(dict)
        self.new_task_names = defaultdict(dict)

        self.latest_added_tasks = defaultdict(dict)
        self.latest_removed_tasks = defaultdict(dict)

        self.is_init = False

    def calculate_dict_differences(
        self, old: dict[str, dict], new: dict[str, dict]
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """
        Compute the differences between two dict[group][expr_no] mappings:
        - added: entries in new but not in old
        - removed: entries in old but not in new
        """
        added = {
            group: {key: new[group][key] for key in set(new[group]) - set(old.get(group, {}))}
            for group in new
        }
        removed = {
            group: {key: old[group][key] for key in set(old[group]) - set(new.get(group, {}))}
            for group in old
        }
        # Also handle complete removal of a group.
        removed.update({group: old[group] for group in old if group not in new})
        return added, removed

    def calculate_task_differences(self):
        return self.calculate_dict_differences(self.old_task_names, self.new_task_names)

    def calculate_active_differences(self):
        # Convert active_tasks to a group-to-expr_no mapping.
        active_map = defaultdict(dict)
        for group, tasks in self.active_tasks.items():
            for task in tasks:
                active_map[group][task.expr_no] = task

        return self.calculate_dict_differences(active_map, self.new_task_names)

    def get_latest_task_changes(self) -> dict:
        return self.latest_added_tasks, self.latest_removed_tasks

    def update(
        self, msg: dict, task_group_flag_updated: defaultdict, task_group_over_expect: DefaultDict[str, dict]
    ) -> dict:
        if not self.is_init:
            self.is_init = True
            self.old_task_names = self.db.find_old_task_names()

        self.logger.debug("Updating all task information")

        # Schedule all tasks together.
        self.new_task_names = defaultdict(dict)
        for one_expr in msg["taskList"]:
            if isinstance(one_expr.get("steps"), Iterable):
                self.new_task_names["main_group"].update(
                    {
                        one_expr["expr_no"]: {
                            "expr_name": one_expr["name"],
                            "expr_steps": [one_step["workstationType"] for one_step in one_expr["steps"]],
                        }
                    }
                )
                self.logger.debug("main_group updated normally")
            else:
                self.logger.info_red_bright(
                    f"Task {one_expr['name']} has incomplete step data; skipping it. Check the DMS task configuration."
                )

        self.logger.debug(f"Old task names: {self.old_task_names}")
        self.logger.debug(f"New task names: {self.new_task_names}")

        self.latest_added_tasks, self.latest_removed_tasks = self.calculate_task_differences()
        if any(self.latest_added_tasks.values()) or any(self.latest_removed_tasks.values()):
            ol = {g: [info["expr_name"] for info in ts.values()] for g, ts in self.old_task_names.items()}
            nw = {g: [info["expr_name"] for info in ts.values()] for g, ts in self.new_task_names.items()}
            ad = {
                group: {expr_no: {"expr_name": info["expr_name"]} for expr_no, info in tasks.items()}
                for group, tasks in self.latest_added_tasks.items()
            }
            rm = {
                group: {expr_no: {"expr_name": info["expr_name"]} for expr_no, info in tasks.items()}
                for group, tasks in self.latest_removed_tasks.items()
            }
            self.logger.info_yellow_bright(
                f"Task sequence updated: \nExisting tasks: {ol}\nNew tasks: {nw}\nAdded tasks: {pformat(ad)}\nRemoved tasks: {pformat(rm)}"
            )

            # Update the task group's updated flag.
            for group, tasks in chain(self.latest_added_tasks.items(), self.latest_removed_tasks.items()):
                if tasks:
                    task_group_flag_updated[group] = True

        self.old_task_names = copy.deepcopy(self.new_task_names)

        self.update_tasks()

        for group, tasks in self.active_tasks.items():
            step_earlier_completed = any(task.get_attr("step_earlier_completed").get() for task in tasks)
            step_later_completed = any(task.get_attr("step_later_completed").get() for task in tasks)
            the_earlier_completed_step = [
                task.get_attr("the_earlier_completed_step").get_attr("name").get()
                for task in tasks
                if task.get_attr("step_earlier_completed").get()
            ]
            the_later_completed_step = [
                task.get_attr("the_later_completed_step").get_attr("name").get()
                for task in tasks
                if task.get_attr("step_later_completed").get()
            ]
            task_group_over_expect[group] = {
                "update": step_later_completed or step_earlier_completed,
                "step_later_completed": step_later_completed,
                "step_earlier_completed": step_earlier_completed,
                "the_earlier_completed_step": the_earlier_completed_step,
                "the_later_completed_step": the_later_completed_step,
            }

        self.logger.debug("All task information has been updated.")
        return self.new_task_names

    def update_tasks(self):
        to_add, to_remove = self.calculate_active_differences()

        # Removed.
        for group, tasks in to_remove.items():
            expr_ids = set(tasks.keys())
            self.active_tasks[group] = [t for t in self.active_tasks[group] if t.expr_no not in expr_ids]
            if not self.active_tasks[group]:
                del self.active_tasks[group]

        # Added.
        for group, tasks in to_add.items():
            for expr_no, _ in tasks.items():
                self.active_tasks[group].append(TaskFactory.create_task(self.logger, self.db, expr_no))

        # Updated.
        for group, tasks in self.active_tasks.items():
            for task in tasks:
                task.update(TaskFactory.update_task(self.db, task.expr_no))



    def find_next_steps(self, group, async_ws_ptr, async_makespan):
        while async_ws_ptr <= async_makespan:
            ws_ptr_matched_tasks = self.find_matching_steps(group, async_ws_ptr)
            if any(ws_ptr_matched_tasks.get(key) for key in ws_ptr_matched_tasks):
                break
            async_ws_ptr += 1
        return async_ws_ptr, ws_ptr_matched_tasks

    def find_matching_steps(self, group, ws_ptr) -> dict[str, dict[str, list[Step]]]:
        matching_tasks = defaultdict(lambda: defaultdict(list))

        for task in self.active_tasks[group]:
            next_step = task.get_next_step()
            if next_step:
                ws_code = next_step.get_attr("ws_code_fjspb").get()

                if next_step.get_attr("start_time").get() == ws_ptr:
                    if next_step.get_attr("status").get() in ["not_start"]:
                        matching_tasks["do_start"][ws_code].append(next_step)

                if next_step.get_attr("end_time").get() == ws_ptr:
                    if next_step.get_attr("status").get() not in ["completed"]:
                        matching_tasks["left"][ws_code].append(next_step)
        return matching_tasks
