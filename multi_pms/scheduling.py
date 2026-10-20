import logging
import json
import json5
import time
import threading
import socket
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

import signal
import queue
import collections
import argparse
import shutil
import requests
import asyncio
import sqlite3
import select
from algorithm.utils.tariff import tariff_snapshot, shift_tariff_periods
from datetime import datetime
from zoneinfo import ZoneInfo

from itertools import chain
import matplotlib.pyplot as plt

from flask import Flask, request, make_response, jsonify
from pprint import pprint, pformat

import colorama
from colorama import Fore, Style
from collections.abc import Iterable

import torch

from scheduler.scheduler import Scheduler
from algorithm.utils.green_objective import shift_valley_window_minutes

from obj.db_tools import *
from utils.async_log import AsyncLog
from utils.args import parse_arguments as parse_scheduling_arguments

from rl.utils.my_utils import group_args_into_dict
from rl.utils.args import parse_arguments as parse_rl_arguments, set_torch_params
from rl.ppo_model import Memory, PPO

from obj.step import Step
from obj.task import AllTasks


class TaskScheduling:
    def __init__(self):
        colorama.init()

        # -------------------- Logging configuration --------------------
        self.logger = AsyncLog(self.__class__.__name__)
        # --------------------------------------------------------

        # Parse service arguments separately from RL arguments.
        self.scheduling_args = parse_scheduling_arguments(allow_unknown=True)
        self.yolo = getattr(self.scheduling_args, "yolo", False)
        # Fix service-level X before slow startup steps such as model loading; floor to CP-SAT minute precision.
        self.yolo_start_time_stamp = (
            datetime.now(ZoneInfo("Asia/Shanghai"))
            .replace(second=0, microsecond=0)
            .strftime("%Y-%m-%d %H:%M:%S")
            if self.yolo
            else None
        )
        self.yolo_valley_start_minute = None
        self.yolo_valley_end_minute = None

        # -------------------- Load scheduling configuration --------------------
        schedule_params_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "database", "schedule_params.jsonc"
        )
        with open(schedule_params_path, "r", encoding="utf-8") as f:
            schedule_params = json5.load(f)
        schedule_cfg = schedule_params.get("schedule", {})
        if self.yolo:
            power_cfg = schedule_params.get("power_info", {})
            self.yolo_valley_start_minute, self.yolo_valley_end_minute = shift_valley_window_minutes(
                self.yolo_start_time_stamp,
                power_cfg.get("valley_start", 22),
                power_cfg.get("valley_end", 8),
            )

        self.database_relpath = schedule_cfg.get("database", "/database")
        # args defines database_file defaults so concurrent services can isolate SQLite/JSON files via CLI.
        self.database_file = self.scheduling_args.database_file
        self.model_name = schedule_cfg.get("model_name", "../rl/data/model/latest_model.pt")
        self.database_dir = os.path.dirname(os.path.abspath(__file__)) + self.database_relpath
        self.database_path = os.path.join(self.database_dir, f"{self.database_file}.sqlite")  # Derive the SQLite filename from the database base name.
        self.json_path = os.path.join(self.database_dir, f"{self.database_file}.json")  # Derive the request snapshot JSON filename from the database base name.

        normal_cfg = schedule_cfg.get("normal", {})
        # --------------------------------------------------------

        # -------------------- Load the RL model --------------------
        self.rl_args, self.model, self.memories, self.env_params = self.load_model()
        if self.scheduling_args.debug:
            self.logger.info_red_bright("Debug mode")
            # args defines debug routes/ports to avoid Flask conflicts during parallel debugging.
            flask_route = self.scheduling_args.debug_flask_route
            flask_port = self.scheduling_args.debug_flask_port
        else:
            self.logger.info_green_bright("Normal mode")
            flask_route = normal_cfg.get("flask_route", "/scheduling")
            flask_port = normal_cfg.get("flask_port", 5060)
        # -------------------------------------------------------

        # -------------------- Database configuration --------------------
        self.db: SchedulerDatabase = SchedulerDatabase(
            self.database_path, self.logger, recover=self.scheduling_args.recover
        )
        if self.yolo:
            # --recover reuses the previous service's fixed X; normal startup uses this process's start time.
            saved_params = (
                self.db.fetch_scheduling_params() if self.scheduling_args.recover else {}
            )
            if saved_params.get("yolo") is True:
                saved_start = saved_params.get("yolo_start_time_stamp")
                saved_valley_start = saved_params.get("yolo_valley_start_minute")
                saved_valley_end = saved_params.get("yolo_valley_end_minute")
                if saved_start and saved_valley_start is not None and saved_valley_end is not None:
                    self.yolo_start_time_stamp = str(saved_start)
                    self.yolo_valley_start_minute = int(saved_valley_start)
                    self.yolo_valley_end_minute = int(saved_valley_end)

            # Modify in-memory power_info only; preserve original schedule_params.jsonc windows for future restarts.
            self.db.power_info["valley_start"] = self._minute_to_clock_text(
                self.yolo_valley_start_minute
            )
            self.db.power_info["valley_end"] = self._minute_to_clock_text(
                self.yolo_valley_end_minute
            )
            shift_tariff_periods(self.db.power_info, self.yolo_start_time_stamp)
            self.logger.info(
                f"YOLO service-level fixed X={self.yolo_start_time_stamp}, off-peak period="
                f"{self.db.power_info['valley_start']}–{self.db.power_info['valley_end']}"
            )
        # --------------------------------------------------------

        # -------------------- Scheduler configuration --------------------
        self.scheduler: Scheduler = Scheduler(
            self.logger, self.db, self.model, self.memories, self.env_params
        )
        # ----------------------------------------------------------

        # Read the current method from CLI arguments parsed by multi_pms.utils.args.parse_arguments.
        self.schedule_method = str(getattr(self.scheduling_args, "schedule_method", "cp_or")).strip().lower()
        # Read the solver time limit from CLI, defaulting to 10 seconds.
        self.schedule_time_limit = getattr(self.scheduling_args, "schedule_time_limit", 10)
        # OR-Tools worker settings; full resolves to all CPU cores available to the process when solving starts.
        self.threads = getattr(self.scheduling_args, "threads", "full")
        self.objective_limit = None  # Request-provided green-objective bound for *_green_limit_* methods.
        self.makespan_limit = None  # Request-provided makespan bound for *_green_limit_strict_* methods.
        self.carbon_limit = None  # Request-provided carbon bound for *_green_limit_strict_* methods.

        self.all_tasks: AllTasks = AllTasks(self.logger, self.db)

        # Received tasks used to detect new arrivals.
        self.new_task_names = {}

        # Each task group has a coroutine with its own updatedFlag and pointer.
        self.task_group_to_coroutine = defaultdict(dict)

        # Task-update flag.
        self.task_group_flag_updated = defaultdict()
        # Unexpected step-completion-time flag.
        self.task_group_over_expect = defaultdict(dict)

        self.msg_lock = threading.Lock()

        # Flask backend receives global DMS state.
        self.app = Flask(__name__)
        self.flask_route = flask_route
        self.flask_port = flask_port
        self.setup_routes()
        self.db.save_scheduling_params(self._collect_scheduling_runtime_params(), overwrite=True)

        self.schedule_id = 0

        # Current outgoing DMS message.
        self.assign_lock = threading.Lock()
        self.assign_cur_queue = queue.Queue()
        self.schedule_status = None  # One-shot scheduling status; reset immediately after sending it to replay.

        # Read configured robots across all sections into robot_list.
        self.robot_list = defaultdict(list)
        robot_list_cfg = schedule_cfg.get("robot_list", {})
        for area, robots in robot_list_cfg.items():
            self.robot_list[area] = list(robots)

        self.exit_event = threading.Event()

        # Coroutine event loop.
        self.async_loop = asyncio.new_event_loop()
        self.trigger_condition = asyncio.Condition()

    def _collect_scheduling_runtime_params(self) -> dict:
        # Persist scheduling.py startup arguments for the frontend and scheduling coroutines.
        runtime_params = {
            "debug": self.scheduling_args.debug,
            "recover": self.scheduling_args.recover,
            "database_file": self.database_file,
            "database_relpath": self.database_relpath,
            "database_dir": self.database_dir,
            "database_path": self.database_path,
            "json_path": self.json_path,
            "model_name": self.model_name,
            "debug_flask_route": self.scheduling_args.debug_flask_route,
            "debug_flask_port": self.scheduling_args.debug_flask_port,
            "flask_route": self.flask_route,
            "flask_port": self.flask_port,
            "schedule_method": self.schedule_method,
            "schedule_time_limit": self.schedule_time_limit,
            "threads": self.threads,
        }
        if self.yolo:
            runtime_params.update(
                {
                    "yolo": True,
                    "yolo_start_time_stamp": self.yolo_start_time_stamp,
                    "yolo_valley_start_minute": self.yolo_valley_start_minute,
                    "yolo_valley_end_minute": self.yolo_valley_end_minute,
                }
            )
        elif self.db.get_scheduling_param("yolo") is True:
            # Write a disabled flag only when replacing an old YOLO startup; ordinary services add no YOLO settings.
            runtime_params["yolo"] = False
        return runtime_params

    @staticmethod
    def _minute_to_clock_text(minute_of_day: int) -> str:
        return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"

    def load_model(self):
        args, device = parse_rl_arguments(allow_unknown=True)
        self.logger.info(f"PyTorch device: {device}")

        nested_args_dict = group_args_into_dict(args, device)
        env_params, model_params, train_params = (
            nested_args_dict["env_params"],
            nested_args_dict["model_params"],
            nested_args_dict["train_params"],
        )

        # Load model weights.
        load_start = time.time()
        memories = Memory()
        model = PPO(model_params, train_params)
        model_ckpt = torch.load(self.model_name, map_location=device, weights_only=True)
        model.policy_old.load_state_dict(model_ckpt, strict=False)  # Allow missing or unexpected checkpoint keys; matching tensor shapes are still required.
        env_params["batch_size"] = args.test_iter
        self.logger.info_green(f"Model loaded in {time.time() - load_start:.2f} seconds")

        return args, model, memories, env_params

    def _collect_access_urls(self) -> tuple[str, list[str]]:
        # Collect localhost and LAN URLs and print them at startup for frontend/API integration.
        local_url = f"http://127.0.0.1:{self.flask_port}"
        lan_urls = []

        candidate_ips = {"127.0.0.1"}
        try:
            candidate_ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                candidate_ips.add(sock.getsockname()[0])
        except OSError:
            pass

        for ip in sorted(candidate_ips):
            if ip and not ip.startswith("127."):
                lan_urls.append(f"http://{ip}:{self.flask_port}")

        return local_url, lan_urls

    def log_access_urls(self):
        # Log local and LAN access URLs for the active service routes.
        local_url, lan_urls = self._collect_access_urls()
        self.logger.info_green_bright(f"Local URL: {local_url}")
        self.logger.info_green_bright(f"Local Status: {local_url}/status")
        self.logger.info_green_bright(f"Local scheduling-method endpoint: {local_url}/schedule_method")

        if lan_urls:
            for lan_url in lan_urls:
                self.logger.info_cyan(f"LAN URL: {lan_url}")
                self.logger.info_cyan(f"LAN Status: {lan_url}/status")
                self.logger.info_cyan(f"LAN scheduling-method endpoint: {lan_url}/schedule_method")
        else:
            self.logger.info_yellow("No usable LAN IPv4 address detected; displaying local URLs only.")

    # Communicate with DMS to receive tasks, equipment, and bottle states.
    def run_flask(self):
        self.logger.info("Thread 1 started: Flask server communicating with DMS to receive task, equipment, and bottle states and return current assignments")
        self.log_access_urls()
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.CRITICAL)
        self.app.run(host="0.0.0.0", port=self.flask_port, debug=True, use_reloader=False)
        self.logger.info("Thread 1 exiting...")

    def task_trigger(self):
        async def _notify():
            async with self.trigger_condition:
                self.trigger_condition.notify_all()  # Notify all waiting coroutines.

        asyncio.run_coroutine_threadsafe(_notify(), self.async_loop)

    def setup_routes(self):
        @self.app.route(self.flask_route, methods=["POST"])
        def scheduling():
            full_data: dict = json.loads(request.data.decode("utf-8"))
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(full_data, f, indent=4, ensure_ascii=False)

            # ========== Read request JSON tariffs before configuration defaults ==========
            # Priority: request JSON > schedule_params.jsonc power_info > defaults.
            if "peak_price" in full_data:
                self.db.power_info["peak_price"] = full_data["peak_price"]
                self.logger.info(f"Peak price updated from request: {full_data['peak_price']}")
            if "valley_price" in full_data:
                self.db.power_info["valley_price"] = full_data["valley_price"]
                self.logger.info(f"Off-peak price updated from request: {full_data['valley_price']}")
            if "makespan_60min_cost_in_valley_price_units" in full_data:
                # Replay may override the valley_price multiplier for 60 minutes of makespan each round.
                self.db.power_info["makespan_60min_cost_in_valley_price_units"] = full_data[
                    "makespan_60min_cost_in_valley_price_units"
                ]
                self.logger.info(
                    "makespan_60min_cost_in_valley_price_units updated from request: "
                    f"{full_data['makespan_60min_cost_in_valley_price_units']}"
                )
            # ========== End tariff update ==========
            def parse_request_limit(field_name):
                raw_value = full_data.get(field_name)
                if raw_value is None:
                    self.logger.info(f"Request does not provide {field_name}.")
                    return None
                try:
                    value = float(raw_value)
                    self.logger.info(f"{field_name} updated from request: {value}")
                    return value
                except (TypeError, ValueError):
                    self.logger.info_yellow(
                        f"Invalid request value for {field_name} ({raw_value}); ignored and set to None."
                    )
                    return None

            self.objective_limit = parse_request_limit("objective_limit")
            self.makespan_limit = parse_request_limit("makespan_limit")
            self.carbon_limit = parse_request_limit("carbon_limit")

            # Update task sequences and the database.
            with self.msg_lock:
                self.db.update_db(full_data)
                self.new_task_names = self.all_tasks.update(
                    full_data, self.task_group_flag_updated, self.task_group_over_expect
                )
                self.update_coroutines(*self.all_tasks.get_latest_task_changes())

            # Trigger coroutine updates.
            batch_group = next(iter(self.new_task_names.keys()), "main_group")
            self.task_trigger()

            # Wait for scheduling to produce DMS instructions.
            time.sleep(2)

            # Instruction tasks.
            assign_list = []
            with self.assign_lock:
                while not self.assign_cur_queue.empty():
                    try:
                        one_assign = self.assign_cur_queue.get_nowait()
                        # Remove assignments from one_assign that are absent from self.new_task_names.
                        all_new_task_names = set().union(*self.new_task_names.values())
                        if one_assign["expr_no"] in all_new_task_names or one_assign["scheduleType"] in [
                            "scheduling_table"
                        ]:
                            assign_list.append(one_assign)
                    except queue.Empty:
                        self.logger.info("assign_queue is currently empty")
                        break

            cur_msg = {
                "code": 200,
                "message": "ok",
                "makespan": self.db.find_makespan_of_all_active_exprs(),
                "algo_cal_time": self.db.find_algo_cal_time(batch_group),
                "schedule_method": self.db.get_schedule_method(self.schedule_method),
                "tariff_info": tariff_snapshot(self.db.power_info),
                "peak_price": self.db.get_power_info("peak_price", 0.6),
                "valley_price": self.db.get_power_info("valley_price", 0.3),
                "makespan_60min_cost_in_valley_price_units": self.db.get_power_info(
                    "makespan_60min_cost_in_valley_price_units", 2.0
                ),
                "beta": self.scheduler.get_last_schedule_metrics(batch_group).get("beta"),
                "solver_objective_value": self.scheduler.get_last_schedule_metrics(batch_group).get(
                    "solver_objective_value"
                ),
                "solver_carbon_objective_scaled": self.scheduler.get_last_schedule_metrics(batch_group).get(
                    "solver_carbon_objective_scaled"
                ),
                "source": self.scheduler.get_last_schedule_metrics(batch_group).get("source"),
                "solver_diagnostics": self.scheduler.get_last_schedule_metrics(batch_group).get(
                    "solver_diagnostics", {}
                ),  # Per-window dynamic-replay diagnostics.
                "green_objective_scale_indicator": self.scheduler.get_last_schedule_metrics(batch_group).get(
                    "scale_indicator"
                ),
                "data": assign_list,
                "all_tasks": self.db.find_all_active_exprs(),
            }
            with self.msg_lock:
                if self.schedule_status == "error":
                    cur_msg["schedule_status"] = "error"  # Return error only when this scheduling round explicitly failed.
                    self.schedule_status = None  # Clear immediately after returning to replay so later successful responses do not inherit failure.

            # pprint("Current outgoing DMS message: ")
            # pprint(cur_msg)

            return jsonify(cur_msg), 200

        @self.app.route("/schedule_method", methods=["GET", "POST"])
        def schedule_method():
            # GET returns the current method; POST selects the method for subsequent scheduling.
            with self.msg_lock:
                if request.method == "POST":
                    payload = request.get_json(silent=True) or {}
                    next_schedule_method = payload.get("schedule_method", payload.get("method"))
                    if not isinstance(next_schedule_method, str) or not next_schedule_method.strip():
                        return (
                            jsonify(
                                {
                                    "code": 400,
                                    "message": "schedule_method must be a non-empty string",
                                }
                            ),
                            400,
                        )

                    self.schedule_method = next_schedule_method.strip().lower()
                    self.db.update_schedule_method(self.schedule_method)
                    self.logger.info_green(f"Scheduling method updated to: {self.schedule_method}")

                current_schedule_method = self.db.get_schedule_method(self.schedule_method).strip().lower()

                return (
                    jsonify(
                        {
                            "code": 200,
                            "message": "ok",
                            "schedule_method": current_schedule_method,
                        }
                    ),
                    200,
                )

        @self.app.route("/status", methods=["GET"])
        def status():
            # /status returns a read-only complete SQLite snapshot without scheduling or database writes.
            with self.msg_lock:
                database_snapshot = self.db.fetch_database_snapshot()
                current_schedule_method = self.db.get_schedule_method(self.schedule_method).strip().lower()

                cur_status = {
                    "code": 200,
                    "message": "ok",
                    "schedule_method": current_schedule_method,
                    "database_file": self.database_file,
                    "flask_route": self.flask_route,
                    "status_route": "/status",
                    "schedule_method_route": "/schedule_method",
                    "tables": database_snapshot,
                }

            return jsonify(cur_status), 200

    def update_coroutines(self, added_tasks, removed_tasks):
        for group, _ in chain(added_tasks.items(), self.new_task_names.items()):
            async_robot_list = self.robot_list["GX305"]

            # Create a new coroutine.
            if group not in self.task_group_to_coroutine or self.task_group_to_coroutine[group].done():
                self.task_group_to_coroutine[group] = self.async_loop.create_task(
                    self.task_tracker(group, async_robot_list)
                )

        # pprint(self.task_group_to_coroutine)

        # Cancel the coroutine when its entire task group is removed.
        for group, _ in removed_tasks.items():
            if group in self.task_group_to_coroutine and not self.new_task_names.get(group, {}):
                self.task_group_to_coroutine.pop(group).cancel()

    def tracker_loop(self):
        self.logger.info("Thread 2 started: coroutine manager")
        asyncio.set_event_loop(self.async_loop)

        async def check_exit_event():
            while not self.exit_event.is_set():
                await asyncio.sleep(1)

            for co in self.task_group_to_coroutine.values():
                co.cancel()

            self.async_loop.stop()

        self.async_loop.create_task(check_exit_event())
        self.async_loop.run_forever()

        self.logger.info("Thread 2 exiting...")

    async def task_tracker(self, group, async_robot_list):
        set_torch_params(device_id=self.rl_args.cuda)
        self.logger.info_green_bright(f"{group} coroutine started...")

        # Tasks matching the current pointer.
        async_ws_ptr_matched_tasks = {}

        # Entering an unchanged task group indicates scheduling recovery.
        if not self.task_group_flag_updated.get(group, False):
            self.logger.info(f"{group} scheduling process restarted; rescheduling is unnecessary...")
            expr_no_list = list(self.new_task_names[group].keys())
            with self.msg_lock:
                async_makespan = self.db.find_makespan(group)
                async_ws_ptr = self.db.find_cur_ptr(group)
            self.logger.info(f"{group} loaded schedule, pointer: {async_ws_ptr}, makespan: {async_makespan}")
            if not async_ws_ptr:
                async_ws_ptr = 0
        else:
            expr_no_list = []
            async_ws_ptr = 0
            async_makespan = 0

        # Record the previous pointer and actual time.
        async_ws_ptr_old = async_ws_ptr
        async_ws_ptr2realtime = time.time()

        while True:
            # Wait for a Flask message to trigger updates.
            async with self.trigger_condition:
                await self.trigger_condition.wait()
            self.logger.info_cyan(f"{group} coroutine triggered; updating...")

            # Count consecutive reschedules without task updates to prevent rescheduling loops.
            rescheule_times = 0
            rescheule_threshold = 1

            # while self.task_group_flag_updated.get(group, False) or self.task_group_over_expect.get(
            #     group, {}
            # ).get("update", False):
            while self.task_group_flag_updated.get(group, False):

                # if not self.task_group_flag_updated.get(group) and self.task_group_over_expect.get(
                #     group, {}
                # ).get("update"):
                #     if rescheule_times >= rescheule_threshold:
                #         self.logger.info_yellow(
                #             f"Too many reschedules without task updates (limit: {rescheule_times}); pause rescheduling at this time"
                #         )
                #         break
                #     rescheule_times += 1

                if self.task_group_flag_updated.get(group):
                    self.logger.info_yellow(f"{group} task sequence updated; rescheduling...")
                # if self.task_group_over_expect.get(group, {}).get("update"):
                #     self.logger.info_yellow(
                #         f"{group} unexpected step durations (early: {self.task_group_over_expect[group].get('step_earlier_completed', False)}, [{self.task_group_over_expect[group].get('the_earlier_completed_step', [])}]; late: {self.task_group_over_expect[group].get('step_later_completed', False)}, [{self.task_group_over_expect[group].get('the_later_completed_step', [])}]); rescheduling..."
                #     )

                with self.msg_lock:
                    # Except during initialization, update expr_no_list only when rescheduling.
                    expr_no_list = list(self.new_task_names.get(group, {}).keys())

                    self.logger.info(f"{group} task-sequence rescheduling started")
                    self.logger.debug(f"Current tasks: {expr_no_list}")

                    calc_start = time.perf_counter()
                    current_schedule_method = self.db.get_schedule_method(self.schedule_method).strip().lower()
                    self.schedule_method = current_schedule_method
                    self.logger.info(f"{group} scheduling method for this round: {current_schedule_method}")
                    async_makespan = self.scheduler.do_schedule(
                        group,
                        async_ws_ptr,
                        expr_no_list,
                        time_limit=self.schedule_time_limit,
                        method=current_schedule_method,
                        objective_limit=self.objective_limit,
                        makespan_limit=self.makespan_limit,
                        carbon_limit=self.carbon_limit,
                        threads=self.threads,
                        yolo=self.yolo,
                    )

                    algo_cal_time = time.perf_counter() - calc_start
                    self.db.update_algo_cal_time(group, algo_cal_time)
                    self.logger.info(f"{group} task-sequence rescheduling completed in {algo_cal_time:.3f} seconds")

                    if async_makespan is not None:
                        self.logger.info(f"{group} new schedule generated, makespan: {async_makespan}")

                        self.all_tasks.update_tasks()
                        self.task_group_flag_updated[group] = False
                        self.task_group_over_expect[group]["update"] = False
                        # Scan the global pointer from zero.
                        async_ws_ptr = 0

                        self.db.update_makespan(group, async_makespan)
                        # Update the schedule Gantt chart.
                        self.scheduler.plot_schedule(
                            async_ws_ptr,
                            expr_no_list,
                            img_name=f".{self.database_relpath}/{group}.svg",
                            yolo=self.yolo,
                        )
                    else:
                        self.logger.info_red_bright(f"{group} FJSPB scheduling failed...")
                        self.schedule_status = "error"  # Mark this scheduling round failed so replay stops repeated requests.
                        await asyncio.sleep(3)

                await asyncio.sleep(0.5)

            try:
                # 0. Find the next batch of tasks.
                with self.msg_lock:
                    self.logger.info(f"{group} searching for the next task batch...")
                    async_ws_ptr, async_ws_ptr_matched_tasks = self.all_tasks.find_next_steps(
                        group, async_ws_ptr, async_makespan
                    )
                    self.db.update_ptr(group, async_ws_ptr)
                    self.db.update_makespan(group, async_makespan)
                    self.logger.info_green_bright(
                        f"{group} found the next task batch, pointer[{async_ws_ptr}], Makespan[{async_makespan}]\n"
                    )

                    # Get the preceding pointer and its actual time.
                    async_ws_ptr_old, async_ws_ptr2realtime = self.db.find_ws_ptr_old_and_realtime(
                        async_ws_ptr
                    )
                    self.logger.info(
                        f"{group} previous pointer: {async_ws_ptr_old}, corresponding actual timestamp: {async_ws_ptr2realtime}, current timestamp: {datetime.now()}"
                    )

                    # Update the plot.
                    self.scheduler.plot_schedule(
                        async_ws_ptr,
                        expr_no_list,
                        img_name=f".{self.database_relpath}/{group}.svg",
                        yolo=self.yolo,
                    )

                await asyncio.sleep(0.1)

                if async_makespan is not None and async_ws_ptr > async_makespan:
                    self.logger.info(f"{group} global pointer traversal complete; all tasks finished...")
                    break

                self.logger.info(
                    f"{group} completing tasks at the current pointer:\n1. Confirm tasks immediately left of the pointer are complete\n2. Start required workstation operations...\n"
                )
                # Complete tasks individually.

                # 1. Confirm all tasks immediately left of the current pointer are complete.
                await asyncio.sleep(0.1)
                left_done = await self.wait_left_of_ptr_tasks_done(
                    group,
                    async_ws_ptr_matched_tasks,
                )
                if left_done:
                    self.logger.info_green(f"{group} all workstation tasks left of the current pointer are complete\n")
                else:
                    self.logger.info(f"{group} workstation tasks left of the current pointer are unfinished; waiting for the next update...\n")
                    continue

                # 2. Start required workstation operations.
                await asyncio.sleep(0.1)
                await self.send_right_of_ptr_tasks_start(
                    group,
                    async_ws_ptr,
                    async_ws_ptr_old,
                    async_ws_ptr2realtime,
                    async_robot_list,
                    async_ws_ptr_matched_tasks,
                )

            except asyncio.CancelledError:
                self.logger.info(f"{group} coroutine cancelled...")
                break
        self.logger.info_yellow_bright(f"{group} coroutine finished...")

    # Wait for all workstation tasks immediately left of the pointer to complete.
    async def wait_left_of_ptr_tasks_done(
        self, group, async_ws_ptr_matched_tasks: dict[str, dict[str, list[Step]]]
    ):
        left_ws_codes = "\n".join(async_ws_ptr_matched_tasks["left"])
        self.logger.info(f"{group} waiting for workstation tasks left of the current pointer to finish: [{left_ws_codes}]")

        done = True
        for ws_code, steps in async_ws_ptr_matched_tasks["left"].items():
            done &= all(step.get_attr("status").get() in ["completed"] for step in steps)

        return done

    # Start required workstation operations.
    async def send_right_of_ptr_tasks_start(
        self,
        group,
        async_ws_ptr,
        async_ws_ptr_old,
        async_ws_ptr2realtime,
        async_robot_list,
        async_ws_ptr_matched_tasks,
    ):
        self.logger.info(f"{group} processing workstations requiring start...")
        for ws_code, steps in dict(async_ws_ptr_matched_tasks["do_start"]).items():
            await asyncio.sleep(0.05)

            self.logger.info(f"{group} sending start to workstation {ws_code}...")

            with self.msg_lock:
                self.schedule_id += 1
                self.logger.info(
                    f"{group} sending the next task to DMS, action: {Fore.GREEN+Style.BRIGHT+'start'+Style.RESET_ALL}"
                )

                for step in steps:
                    with self.assign_lock:

                        # Select a robot.
                        if "labx_303" in self.db.fetch_ws_sections(step.get_attr("ws_code_fjspb").get()):
                            exec_robot = async_robot_list[0]
                        else:
                            exec_robot = async_robot_list[1]

                        allow_send = True
                        # Compare expected pointer-gap minutes against the actual elapsed time for this step.
                        # if async_ws_ptr_old is not None and async_ws_ptr2realtime is not None:
                        #     realtime_delta_min = (
                        #         datetime.now() - datetime.strptime(async_ws_ptr2realtime, "%Y-%m-%d %H:%M:%S")
                        #     ).total_seconds() / 60.0
                        #     ptr_delta_min = async_ws_ptr - async_ws_ptr_old
                        #     self.logger.info(
                        #         f"ptr_delta_min: {ptr_delta_min}, realtime_delta_min: {realtime_delta_min}"
                        #     )

                        #     # If actual elapsed time is shorter than the pointer gap, defer start until the next scheduling update.
                        #     if realtime_delta_min < ptr_delta_min * 1.0:
                        #         allow_send = False

                        if allow_send:
                            one_assignment = step.do_start(self.schedule_id, exec_robot)
                            self.assign_cur_queue.put(one_assignment)
                        else:
                            self.logger.info_yellow(
                                f"Actual elapsed time since the previous pointer is shorter than the pointer gap; defer start for workstation {step.get_attr('ws_code_fjspb').get()} until the next scheduling update..."
                            )
                    self.logger.info(pformat(one_assignment))
        self.logger.info(f"{group} started all required workstations...\n")

    # Handle Ctrl-C.
    def signal_handler(self, sig, frame):
        # self.logger.info("\nExiting gracefully...")
        self.exit_event.set()

    def run(self):
        self.thread_async_loop = threading.Thread(target=self.tracker_loop)
        self.thread_async_loop.start()

        time.sleep(0.5)

        self.thread_connect_dms = threading.Thread(target=self.run_flask, daemon=True)
        self.thread_connect_dms.start()

        signal.signal(signal.SIGINT, self.signal_handler)
        while not self.exit_event.is_set():
            time.sleep(0.5)

        self.logger.info("Main program finished; closing the database...")


if __name__ == "__main__":
    TS = TaskScheduling()
    TS.run()
