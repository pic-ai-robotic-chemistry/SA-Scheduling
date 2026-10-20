import json
import time
import sys, os
import sqlite3
import threading

import colorama
import json5
from colorama import Fore, Style

from collections import defaultdict
from pprint import pprint
from datetime import datetime
from zoneinfo import ZoneInfo

from typing import Any

from multi_pms.utils.async_log import AsyncLog
from algorithm.utils.tariff import load_tariff_config, tariff_snapshot

colorama.init()


class SchedulerDatabase:
    def __init__(self, db_path: str, logger: AsyncLog, recover=False, *args):
        self.db_path = db_path
        self.logger = logger
        self.lock = threading.RLock()

        if not recover:
            self.is_db_init = False
        else:
            self.is_db_init = not self._is_empty_db()

        # Load scheduling settings from schedule_params.jsonc; retain power_info for existing callers.
        schedule_params_path = os.path.join(os.path.dirname(self.db_path), "schedule_params.jsonc")
        with open(schedule_params_path, "r", encoding="utf-8") as f:
            self.schedule_params = json5.load(f)
        self.power_info = load_tariff_config(self.schedule_params.get("power_info", {}), os.path.dirname(schedule_params_path))

    def get_power_info(self, key: str, default: Any = None):
        return self.power_info.get(key, default)

    def _resolve_ws_power_watts(self, ws_type: str, ws_version: str) -> int:
        # Power keys may include a readable suffix: workstationTypeCode-definitionVersion-readableName.
        # Match only the workstationTypeCode-definitionVersion prefix; unconfigured stations default to 0 W.
        power_map = self.power_info.get("power_info", {})
        composite_prefix = f"{ws_type}-{ws_version}"
        power_value = next(
            (
                value
                for key, value in power_map.items()
                if key == composite_prefix or key.startswith(f"{composite_prefix}-")
            ),
            0,
        )
        try:
            return int(power_value)
        except (TypeError, ValueError):
            return 0

    # Create a separate connection for each operation.
    def _connect(self):
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _string_id(value: Any) -> str | None:

        return None if value is None else str(value)

    # Private helper: check whether the database has no tables.
    def _is_empty_db(self) -> bool:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            # Ignore the parameter table when detecting initialized scheduling tables for recovery.
            cursor.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                AND name NOT IN ('sqlite_sequence', 'scheduling_param_tb')
                """
            )
            tables = cursor.fetchall()
            return not tables

    def fetch_database_snapshot(self) -> dict[str, dict[str, Any]]:
        # Export a complete SQLite snapshot by table for read-only endpoints such as /status.
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name != 'sqlite_sequence'
                ORDER BY name
                """
            )
            table_names = [row[0] for row in cursor.fetchall()]

            snapshot = {}
            for table_name in table_names:
                cursor.execute(f"SELECT * FROM {table_name}")
                columns = [description[0] for description in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                snapshot[table_name] = {
                    "columns": columns,
                    "row_count": len(rows),
                    "rows": rows,
                }

            return snapshot

    # Clear and initialize database tables.
    def reset_db(self):
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            scheduling_params = self.fetch_scheduling_params()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            for table in tables:
                if table[0] != "sqlite_sequence":
                    cursor.execute(f"DROP TABLE {table[0]}")
            conn.commit()

            # Workstation definitions.
            cursor.execute(
                """CREATE TABLE ws_defi_info (
                            workstationTypeCode TEXT PRIMARY KEY,
                            version TEXT,
                            allocates TEXT,
                            commands TEXT,
                            pipelines TEXT
                        )"""
            )
            # Workstation instances.
            cursor.execute(
                """CREATE TABLE ws_info (
                            workstationTypeCode TEXT,
                            workstationCode TEXT PRIMARY KEY,
                            version TEXT,
                            workstationName TEXT,
                            powerWatts INTEGER,
                            status TEXT,
                            sectionList TEXT,
                            allocates TEXT,
                            capacity INTEGER 
                        )"""
            )
            # Robots.
            cursor.execute(
                """CREATE TABLE robot_info (
                            robotTypeCode TEXT,
                            robotTypeName TEXT,
                            robotCode TEXT PRIMARY KEY,
                            robotName TEXT,
                            status TEXT,
                            electricityQuantity INTEGER,
                            sectionList TEXT,
                            clampList TEXT,
                            containerInfos TEXT
                        )"""
            )
            # Tasks.
            cursor.execute(
                """CREATE TABLE task_info (
                            task_group TEXT,
                            expr_id TEXT,
                            expr_no TEXT PRIMARY KEY,
                            expr_name TEXT,
                            status TEXT,
                            stamp TEXT,
                            is_scheduled BOOLEAN DEFAULT FALSE,
                            abstract_status TEXT
                        )"""
            )
            # Task schedule; start_time defaults to the maximum value.
            cursor.execute(
                f"""CREATE TABLE task_scheduled (
                            task_group TEXT,
                            name TEXT,
                            expr_name TEXT,
                            expr_no TEXT,
                            expr_id TEXT,
                            stamp TEXT,

                            step_id TEXT,
                            job_length INTEGER,
                            
                            ws_type TEXT,
                            ws_version TEXT,
                            ws_arr TEXT,
                            ws_code_fjspb TEXT,
                            next_step_ws_code_fjspb TEXT,
                            time INTEGER,
                            
                            fjspb_index INTEGER,
                            status TEXT,
                            real_start_time TEXT,
                            real_end_time TEXT,
                            start_time INTEGER DEFAULT {sys.maxsize},
                            end_time INTEGER,
                            duration INTEGER,
                            lock BOOLEAN DEFAULT FALSE,
                            lock_periods TEXT, 

                            containers TEXT,
                            pipeline TEXT,
                            
                            PRIMARY KEY (expr_no, fjspb_index)
                        )"""
            )

            # Time pointers.
            cursor.execute(
                """CREATE TABLE ptr_tb (
                            task_group TEXT PRIMARY KEY,
                            ptr INTEGER DEFAULT 0,
                            makespan INTEGER,
                            algo_cal_time REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )"""
            )

            self._create_scheduling_param_table(cursor)
            self._upsert_scheduling_params(cursor, scheduling_params, overwrite=True)

            conn.commit()

    def _create_scheduling_param_table(self, cursor):
        # Runtime settings table stores startup arguments and the dynamically selectable schedule_method.
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS scheduling_param_tb (
                        param_name TEXT PRIMARY KEY,
                        param_value TEXT,
                        param_type TEXT,
                        updated_at TEXT
                    )"""
        )

    @staticmethod
    def _shanghai_now_text() -> str:
        # Use Shanghai time for parameter updates regardless of the server timezone.
        return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _serialize_scheduling_param(value: Any) -> tuple[str, str]:
        # Serialize settings consistently so /status and scheduling restore original types.
        if isinstance(value, bool):
            return json.dumps(value), "bool"
        if isinstance(value, int):
            return json.dumps(value), "int"
        if isinstance(value, float):
            return json.dumps(value), "float"
        if value is None:
            return "null", "none"
        return str(value), "str"

    @staticmethod
    def _deserialize_scheduling_param(value: str | None, value_type: str | None) -> Any:
        # SQLite stores parameter values as text; restore startup types where possible.
        if value is None:
            return None
        try:
            if value_type == "bool":
                return bool(json.loads(value))
            if value_type == "int":
                return int(json.loads(value))
            if value_type == "float":
                return float(json.loads(value))
            if value_type == "none":
                return None
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
        return value

    def _upsert_scheduling_params(self, cursor, params: dict[str, Any], overwrite: bool = True):
        # Shared by reset_db and public writers; overwrite=False fills missing settings only.
        updated_at = self._shanghai_now_text()
        for param_name, raw_value in params.items():
            param_value, param_type = self._serialize_scheduling_param(raw_value)
            if overwrite:
                cursor.execute(
                    """
                    INSERT INTO scheduling_param_tb (param_name, param_value, param_type, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(param_name) DO UPDATE SET
                        param_value = excluded.param_value,
                        param_type = excluded.param_type,
                        updated_at = excluded.updated_at
                    """,
                    (param_name, param_value, param_type, updated_at),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO scheduling_param_tb (param_name, param_value, param_type, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (param_name, param_value, param_type, updated_at),
                )

    def ensure_scheduling_param_table(self):
        # Create settings before the first DMS request so the method is persisted rather than memory-only.
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            self._create_scheduling_param_table(cursor)
            conn.commit()

    def save_scheduling_params(self, params: dict[str, Any], overwrite: bool = True):
        # Persist runtime arguments read at scheduling.py startup.
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            conn.execute("BEGIN")
            self._create_scheduling_param_table(cursor)
            self._upsert_scheduling_params(cursor, params, overwrite=overwrite)
            conn.commit()

    def fetch_scheduling_params(self) -> dict[str, Any]:

        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = 'scheduling_param_tb'
                """
            )
            if cursor.fetchone() is None:
                return {}

            cursor.execute("SELECT param_name, param_value, param_type FROM scheduling_param_tb")
            return {
                name: self._deserialize_scheduling_param(value, value_type)
                for name, value, value_type in cursor.fetchall()
            }

    def get_scheduling_param(self, param_name: str, default: Any = None) -> Any:
        # Read one setting for the frontend and scheduler.
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            self._create_scheduling_param_table(cursor)
            cursor.execute(
                """
                SELECT param_value, param_type
                FROM scheduling_param_tb
                WHERE param_name = ?
                """,
                (param_name,),
            )
            row = cursor.fetchone()
            if row is None:
                return default
            return self._deserialize_scheduling_param(row[0], row[1])

    def get_schedule_method(self, default: str = "cp_or") -> str:
        # Read the current method from the database before scheduling.
        schedule_method = self.get_scheduling_param("schedule_method", default)
        return str(schedule_method or default)

    def update_schedule_method(self, schedule_method: str):
        # Used by the frontend POST endpoint to select the next scheduling method.
        self.save_scheduling_params({"schedule_method": schedule_method}, overwrite=True)

    # Update database workstation states on a DMS message.
    def update_db(self, msg: dict[str, Any]):
        task_list, robot_list, ws_list, ws_defi_list = (
            msg["taskList"],
            msg["robotList"],
            msg["workstationList"],
            msg["workstationDefinitionList"],
        )

        # Reset the database if tasks have not been initialized.
        if not self.is_db_init:
            self.reset_db()

        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            try:
                # ---------------------- Update workstation definitions ----------------------
                conn.execute("BEGIN")
                for ws_defi in ws_defi_list:
                    cursor.execute(
                        "INSERT OR REPLACE INTO ws_defi_info (workstationTypeCode, version, allocates, commands, pipelines) VALUES (?, ?, ?, ?, ?)",
                        (
                            ws_defi["workstationTypeCode"],
                            ws_defi["version"],
                            json.dumps(ws_defi["allocates"]),
                            json.dumps(ws_defi["commands"]),
                            json.dumps(ws_defi["pipelines"]),
                        ),
                    )
                # Insert the virtual ProcessControl workstation separately.
                cursor.execute(
                    "INSERT OR REPLACE INTO ws_defi_info (workstationTypeCode, version, allocates, commands, pipelines) VALUES (?, ?, ?, ?, ?)",
                    (
                        "ProcessControl",
                        "1.0",
                        "",
                        "",
                        "",
                    ),
                )
                conn.commit()
                # -----------------------------------------------------------
                # ---------------------- Update workstation instances ----------------------
                conn.execute("BEGIN")
                for ws in ws_list:
                    power_watts = self._resolve_ws_power_watts(
                        ws["workstationTypeCode"], ws["version"]
                    )
                    cursor.execute(
                        "INSERT OR REPLACE INTO ws_info (workstationTypeCode, workstationCode, version, workstationName, powerWatts, status, sectionList, allocates, capacity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            ws["workstationTypeCode"],
                            ws["workstationCode"],
                            ws["version"],
                            ws["workstationName"],
                            power_watts,
                            ws["status"],
                            json.dumps(ws["sectionList"]),
                            json.dumps(ws["allocates"]),
                            ws.get("capacity", 1),  # Default capacity is 1, processing one step at a time.

                        ),
                    )
                # Insert virtual workflow-control workstations separately.
                # Use capacity 1e3 for all virtual workflow stations to avoid scheduling bottlenecks.
                virtual_ws_capacity = int(1e3)
                for pc_ws_code in ["start", "end"]:
                    cursor.execute(
                        "INSERT OR REPLACE INTO ws_info (workstationTypeCode, workstationCode, version, workstationName, powerWatts, status, sectionList, allocates, capacity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            "ProcessControl",
                            pc_ws_code,
                            pc_ws_code,
                            pc_ws_code,
                            0,
                            "IDLE",
                            "",
                            "",
                            virtual_ws_capacity,
                        ),
                    )
                conn.commit()
                # -------------------------------------------------------------------
                # ---------------------- Update robots ----------------------
                conn.execute("BEGIN")
                for rb in robot_list:
                    cursor.execute(
                        "INSERT OR REPLACE INTO robot_info (robotTypeCode, robotTypeName, robotCode, robotName, status, electricityQuantity, sectionList, clampList, containerInfos) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            rb["robotTypeCode"],
                            rb["robotTypeName"],
                            rb["robotCode"],
                            rb["robotName"],
                            rb["status"],
                            rb["electricityQuantity"],
                            json.dumps(rb["sectionList"]),
                            json.dumps(rb["clampList"]),
                            json.dumps(rb["containerInfos"]),  # Default capacity is 1 (one step per task).
                        ),
                    )
                conn.commit()
                # -------------------------------------------------------------------

                # ---------------------- Update tasks ----------------------
                conn.execute("BEGIN")
                for exp in task_list:
                    expr_id = self._string_id(exp["id"])
                    cursor.execute(
                        """
                        INSERT INTO task_info (expr_id, expr_no, expr_name, status, stamp) 
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(expr_no) DO UPDATE SET
                            expr_id = excluded.expr_id,
                            expr_name = excluded.expr_name,
                            status = excluded.status,
                            stamp = excluded.stamp
                        """,
                        (expr_id, exp["expr_no"], exp["name"], exp["status"], exp["stamp"]),
                    )

                # Then process tasks absent from task_list or with exp[status]='error'.
                expr_no_active_list = [exp["expr_no"] for exp in task_list if exp["status"] != "error"]

                # 1. Set abstract_status = ACTIVE_STATUS for every non-error task.
                if expr_no_active_list:
                    placeholders_active = ",".join("?" * len(expr_no_active_list))
                    cursor.execute(
                        f"""
                        UPDATE task_info
                        SET abstract_status = ?
                        WHERE expr_no IN ({placeholders_active})
                        """,
                        ("active", *expr_no_active_list),
                    )

                # 2. Handle remaining tasks, including errors in task_list and task_info tasks absent from task_list.
                if expr_no_active_list:
                    placeholders_not_active = ",".join("?" * len(expr_no_active_list))
                    cursor.execute(
                        f"""
                        UPDATE task_info
                        SET abstract_status = ?, is_scheduled = ?
                        WHERE expr_no NOT IN ({placeholders_not_active})
                        """,
                        ("inactive", False, *expr_no_active_list),
                    )
                else:
                    # If there are no active tasks, update all tasks.
                    cursor.execute(
                        """
                        UPDATE task_info
                        SET abstract_status = ?, is_scheduled = ?
                        """,
                        ("inactive", False),
                    )

                conn.commit()
                # -----------------------------------------------------------
                # ---------------------- Parse the task list ----------------------
                if len(task_list) > 0:
                    conn.execute("BEGIN")

                    for exp in task_list:
                        expr_name = exp["name"]
                        expr_id = self._string_id(exp["id"])

                        # Null durations denote untimed actions such as injection or aspiration; use 1 for scheduling.
                        fjspb_index = 0
                        is_lock = False
                        for step in exp["steps"]:
                            if step["workstationType"] == "lock":
                                is_lock = True
                            if step["workstationType"] == "unlock":
                                is_lock = False
                            if step["workstationType"] not in ["lock", "unlock"]:
                                ws_type = step["workstationType"]
                                ws_version = (
                                    ws_type if ws_type in ["start", "end"] else step["definitionVersion"]
                                )
                                ws_arr = self.fetch_ws_by_type_and_version(ws_type, ws_version)

                                step_time = int(step["time"]) if int(step["time"]) != 0 else 1
                                step_id = self._string_id(step["id"])

                                cursor.execute(
                                    """
                                    INSERT OR IGNORE INTO task_scheduled 
                                    (expr_no, expr_id, fjspb_index, expr_name, stamp, step_id, ws_type, ws_version, ws_arr, ws_code_fjspb, time, lock) 
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        exp["expr_no"],
                                        expr_id,
                                        fjspb_index,
                                        expr_name,
                                        exp["stamp"],
                                        step_id,
                                        ws_type,
                                        ws_version,
                                        ws_arr,
                                        None,
                                        step_time,
                                        is_lock,
                                    ),
                                )
                                fjspb_index += 1

                    # Update job_length in the task table.
                    cursor.execute(
                        """ SELECT expr_no, MAX(fjspb_index) FROM task_scheduled GROUP BY expr_no """
                    )
                    max_fjspb_indexes = cursor.fetchall()
                    for expr_no, max_fjspb_index in max_fjspb_indexes:
                        cursor.execute(
                            """UPDATE task_scheduled SET job_length = ? WHERE expr_no = ?""",
                            (max_fjspb_index + 1, expr_no),
                        )

                    conn.commit()

                    # Initialize tasks that have not yet been initialized.
                    if not self.is_db_init:
                        self.logger.info_green_bright("Received tasks while the task list was empty; initializing the scheduling database...")
                        self.is_db_init = True

                # ---------------------------------------------------------

                # ---------------------- Update task records ----------------------
                conn.execute("BEGIN")
                if len(task_list) > 0:
                    for exp in task_list:
                        fjspb_index = 0
                        for step in exp["steps"]:
                            if step["workstationType"] not in ["lock", "unlock"]:
                                if "startTime" not in step:
                                    self.logger.warning(
                                        f"step is missing 'startTime' key (expr_no={exp['expr_no']})"
                                    )

                                cursor.execute(
                                    """
                                    UPDATE task_scheduled 
                                    SET status = ?, real_start_time = ?, real_end_time = ?
                                    WHERE expr_no = ? AND fjspb_index = ?
                                    """,
                                    (
                                        step["status"],
                                        step.get("startTime"),
                                        step["endTime"],
                                        exp["expr_no"],
                                        fjspb_index,
                                    ),
                                )
                                fjspb_index += 1
                conn.commit()
                # ---------------------------------------------------------

                # Set start_time=sys.maxsize in task_scheduled for all inactive task_info tasks.
                # ---------------------- Reschedule when new tasks arrive ----------------------
                conn.execute("BEGIN")
                cursor.execute("SELECT expr_no FROM task_info WHERE abstract_status = ?", ("inactive",))
                inactive_expr_nos = [row[0] for row in cursor.fetchall()]
                cursor.execute(
                    f"""
                    UPDATE task_scheduled
                    SET start_time = ?
                    WHERE expr_no IN ({','.join('?' * len(inactive_expr_nos))})
                    """,
                    (sys.maxsize, *inactive_expr_nos),
                )
                conn.commit()
                # ---------------------------------------------------------

                # ---------------------- Update timestamps ----------------------
                # ---------------------------------------------------------

            except sqlite3.Error as e:
                conn.rollback()
                print(f"Function {self.update_db.__name__} raised an error: {e}; transaction rolled back...")

        return

    # Read all workstation codes from the database.
    def find_ws_code_list_from_db(self) -> list:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(f"SELECT workstationCode, workstationTypeCode, sectionList FROM ws_info")
            ws_codes = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
            return ws_codes

    @staticmethod
    def _parse_stamp(stamp_value):
        return datetime.strptime(stamp_value, "%Y-%m-%d %H:%M:%S")

    # Build data only for expr_no_list tasks with start_time >= cur_ptr.
    def construct_fespb_jobs_data_from_db(self, cur_ptr: int, expr_no_list: list, is_plot=False) -> dict:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            expr_no_placeholders = ",".join("?" * len(expr_no_list))

            if is_plot:
                # For plotting.
                cursor.execute(
                    f"""
                    SELECT * FROM task_scheduled 
                    WHERE expr_no IN ({expr_no_placeholders}) AND start_time >= ?
                    ORDER BY fjspb_index
                    """,
                    (*expr_no_list, cur_ptr),
                )
            else:
                # Select unfinished steps starting at or after cur_ptr, including not_start/in_progress/error.
                cursor.execute(
                    f"""
                    SELECT * FROM task_scheduled 
                    WHERE expr_no IN ({expr_no_placeholders}) AND start_time >= ? AND status != 'completed'
                    ORDER BY fjspb_index
                    """,
                    (*expr_no_list, cur_ptr),
                )

            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

            cursor.execute(
                f"""
                SELECT stamp
                FROM task_info
                WHERE expr_no IN ({expr_no_placeholders})
                """,
                expr_no_list,
            )
            stamp_values = [self._parse_stamp(row[0]) for row in cursor.fetchall()]
            sim_cur_time_stamp = min(stamp_values).strftime("%Y-%m-%d %H:%M:%S")

            jobs_data = {
                "detail": defaultdict(list),
                "info": {
                    "machine_capacities": {},
                    "machine_power_info": {},
                    "num_jobs": None,
                    "num_machines": None,
                    "num_opers": None,
                    "machines_available_time": {},
                    "job_avail_time": {},
                    "job_lock": {},
                    "job_lock_periods": {},
                    "job_rightnow": {},
                },
            }

            # Step lock flags for each task.
            job_lock = defaultdict(list)

            # Read tasks to schedule and all their steps.
            for step_data in rows:
                expr_no = step_data["expr_no"]

                if (
                    not is_plot
                    and step_data["status"] not in ["not_start"]
                    and step_data["ws_code_fjspb"] is not None
                ):


                    # Do not reschedule in-progress steps.
                    continue
                else:
                    ws_arr_list = step_data["ws_arr"].split(",")

                    # Detect operations without eligible machines (ws_arr empty or containing only empty strings).
                    ws_arr_filtered = [w for w in ws_arr_list if w]
                    if not ws_arr_filtered:
                        self.logger.info_red_bright(
                            f"[Diagnostic] Operation has no available workstations! expr_no={step_data['expr_no']}, "
                            f"fjspb_index={step_data['fjspb_index']}, "
                            f"ws_type={step_data.get('ws_type', '?')}, "
                            f"ws_version={step_data.get('ws_version', '?')}, "
                            f"ws_arr='{step_data['ws_arr']}'"
                        )

                    job_lock[expr_no].append(step_data["lock"])
                    jobs_data["detail"][expr_no].append(
                        (
                            ws_arr_filtered,
                            [step_data["time"]] * len(ws_arr_filtered),
                            "",  # Placeholder parameters.
                            {
                                "start_time": step_data["start_time"],
                                "end_time": step_data["end_time"],
                                "ws_code_fjspb": step_data["ws_code_fjspb"],
                                "fjspb_index": step_data["fjspb_index"],
                                "job_length": step_data["job_length"],
                                "lock": step_data["lock"],
                                "status": step_data["status"],
                            },
                        )
                    )

            jobs_data["info"].update(
                {
                    "num_jobs": len(jobs_data["detail"]),
                    "num_opers": sum(len(v) for v in jobs_data["detail"].values()),
                }
            )

            machines_list = set()
            for one_task in jobs_data["detail"].values():
                for machines, *_ in one_task:
                    machines_list.update(machines)
            machines_list = list(machines_list)

            jobs_data["info"]["num_machines"] = len(machines_list)
            jobs_data["info"]["machine_capacities"].update(
                self.find_capacity_by_ws_code_from_db(machines_list)
            )
            jobs_data["info"]["machine_power_info"].update(self.find_power_by_ws_code_from_db(machines_list))

            # Scan tasks to determine all workstation availability times at cur_ptr.
            # If start_time > cur_ptr or end <= cur_ptr, ws_code_fjspb is available at cur_ptr.
            # If start_time < cur_ptr < end, ws_code_fjspb is available at end.
            # If start_time == cur_ptr:
            #   1. For not_start, ws_code_fjspb is available at cur_ptr.
            #   2. Otherwise it is available at end.
            # For repeated ws_code_fjspb entries, use the maximum availability time.
            eligible_machines_list = list(jobs_data["info"]["machine_capacities"].keys())
            machines_available_time = defaultdict(lambda: cur_ptr)

            cursor.execute(
                f"""
                SELECT * FROM task_scheduled 
                WHERE expr_no IN ({expr_no_placeholders})
                ORDER BY fjspb_index
                """,
                expr_no_list,
            )
            columns = [description[0] for description in cursor.description]

            # Earliest start time of each task.
            job_avail_time = defaultdict(int)
            # Lock intervals of each task.
            job_lock_periods = defaultdict(list)

            for row in cursor.fetchall():
                step_data = dict(zip(columns, row))
                expr_no = step_data["expr_no"]
                ws_code_fjspb = step_data["ws_code_fjspb"]
                start_time = step_data["start_time"]
                end_time = step_data["end_time"]
                lock_periods = step_data["lock_periods"]

                job_lock_periods[expr_no] = json.loads(lock_periods) if lock_periods else []
                ws_arr_list = step_data["ws_arr"].split(",")

                for ws_code in ws_arr_list:


                    if (
                        step_data["name"] is not None and end_time is None
                    ):  # New, unassigned task without ws_code_fjspb.
                        if ws_code in eligible_machines_list:
                            machines_available_time[ws_code] = max(machines_available_time[ws_code], cur_ptr)
                        job_avail_time[expr_no] = max(job_avail_time[expr_no], cur_ptr)
                    elif ws_code == ws_code_fjspb:
                        # For tasks starting after cur_ptr or ending before it, use cur_ptr.
                        # For tasks still running at cur_ptr, use end_time.
                        # Handle two cases when the task starts exactly at cur_ptr.
                        if start_time > cur_ptr or end_time <= cur_ptr:
                            available_time = cur_ptr
                        elif start_time < cur_ptr < end_time:
                            available_time = end_time
                        else:
                            if step_data["status"] in ["not_start"]:
                                available_time = cur_ptr
                            else:
                                available_time = end_time

                        if ws_code in eligible_machines_list:
                            machines_available_time[ws_code] = max(
                                machines_available_time[ws_code], available_time, cur_ptr
                            )
                        job_avail_time[expr_no] = max(job_avail_time[expr_no], available_time, cur_ptr)
                    else:
                        if ws_code in eligible_machines_list:
                            machines_available_time[ws_code] = max(machines_available_time[ws_code], cur_ptr)
                        job_avail_time[expr_no] = max(job_avail_time[expr_no], cur_ptr)

            # If cur_ptr lies inside lock_periods, require this job to start exactly at job_avail_time.
            # job_avail_time[expr_no] != end prevents falsely joining an unlocked operation to the last locked operation.
            job_rightnow = {
                expr_no: any(
                    start <= cur_ptr < end and job_avail_time[expr_no] != end for start, end in periods
                )
                for expr_no, periods in job_lock_periods.items()
            }

            jobs_data["info"].update(
                {
                    "machines_available_time": machines_available_time,
                    "job_avail_time": job_avail_time,
                    "job_lock": job_lock,
                    "job_lock_periods": job_lock_periods,
                    "job_rightnow": job_rightnow,
                    "cur_time_stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sim_cur_time_stamp": sim_cur_time_stamp,
                    **tariff_snapshot(self.power_info),
                    "peak_price": self.power_info.get("peak_price", 0.6),
                    "valley_price": self.power_info.get("valley_price", 0.3),
                    "valley_start": self.power_info.get("valley_start", 22),
                    "valley_end": self.power_info.get("valley_end", 8),
                    # Makespan time value in the green objective:
                    # X means 60 minutes of makespan are valued at approximately X * valley_price yuan.
                    "makespan_60min_cost_in_valley_price_units": self.power_info.get(
                        "makespan_60min_cost_in_valley_price_units", 2.0
                    ),
                }
            )


            return jobs_data

    def find_finite_step_end_time(self, expr_no: str, fjspb_index: int) -> int | None:
        """Query finite end times for persisted steps to fill the preceding-operation constraint for virtual end steps across windows."""
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT end_time
                FROM task_scheduled
                WHERE expr_no = ?
                  AND fjspb_index = ?
                  AND start_time < ?
                  AND end_time IS NOT NULL
                  AND end_time < ?
                """,
                (expr_no, fjspb_index, sys.maxsize, sys.maxsize),
            )
            result = cursor.fetchone()
            return int(result[0]) if result is not None else None

    # Read workstation capacities by ws_code.
    def find_capacity_by_ws_code_from_db(self, codes: list[str]) -> dict[str, int]:
        if not codes:
            return {}

        placeholders = ",".join("?" for _ in codes)
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT workstationCode, capacity FROM ws_info WHERE workstationCode IN ({placeholders})",
                codes,
            )
            result = cursor.fetchall()

            capacities = {code: 0 for code in codes}
            for ws_code, capacity in result:
                capacities[ws_code] = capacity

            return capacities

    # Read workstation power by ws_code.
    def find_power_by_ws_code_from_db(self, codes: list[str]) -> dict[str, int]:
        if not codes:
            return {}

        placeholders = ",".join("?" for _ in codes)
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT workstationCode, powerWatts FROM ws_info WHERE workstationCode IN ({placeholders})",
                codes,
            )
            result = cursor.fetchall()

            powers = {code: 0 for code in codes}
            for ws_code, power in result:
                powers[ws_code] = power

            return powers

    def find_makespan(self, task_group: str | None = None) -> int | None:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            if task_group is None:
                cursor.execute("SELECT max(end_time) FROM task_scheduled")
            else:
                cursor.execute("SELECT makespan FROM ptr_tb WHERE task_group = ?", (task_group,))
            row = cursor.fetchone()
            if row is None:
                return None

            makespan = row[0]
            return makespan

    # Find makespan across active tasks.
    def find_makespan_of_all_active_exprs(self) -> int:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT MAX(end_time) 
                FROM task_scheduled 
                WHERE expr_no IN (
                    SELECT expr_no FROM task_info WHERE abstract_status = ?
                )
                """,
                ("active",),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            makespan = row[0]
            return makespan

    def find_cur_ptr(self, task_group: str) -> int:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT ptr FROM ptr_tb WHERE task_group = ?", (task_group,))
            row = cursor.fetchone()
            if row is None:
                return 0

            cur_ptr = row[0]
            return cur_ptr

    # Find names of all active tasks in task_info.
    def find_all_active_exprs(self) -> list[str]:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT expr_name FROM task_info where abstract_status = ?", ("active",))
            rows = cursor.fetchall()
            expr_names = [row[0] for row in rows]
            return expr_names

    def write_ws_table_to_db(self, task_group: str, expr_no_list: list, ws_table: dict) -> int:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("BEGIN")
            expr_no_placeholders = ",".join("?" for _ in expr_no_list)

            cursor.execute(
                f"""
                    UPDATE task_info
                    SET
                    task_group = ?,
                    is_scheduled = ?
                    WHERE expr_no IN ({expr_no_placeholders})
                """,
                [task_group, True, *expr_no_list],
            )
            conn.commit()

            cursor.execute("BEGIN")
            # Group schedule rows by machine.
            for ws_code_fjspb, tasks in ws_table.items():
                for task in tasks:
                    cursor.execute(
                        """
                        UPDATE task_scheduled
                        SET
                        task_group = ?,
                        name = ?,
                        ws_code_fjspb = ?,
                        start_time = ?,
                        end_time = ?,
                        duration = ?
                        WHERE expr_no = ? AND fjspb_index = ?""",
                        (
                            task_group,
                            task["name"],
                            task["ws_code_fjspb"],
                            task["start_time"],
                            task["end_time"],
                            task["duration"],
                            task["expr_no"],
                            task["fjspb_index"],
                        ),
                    )

            conn.commit()

            # Update next_step_ws_code_fjspb for each task.
            cursor.execute("BEGIN")
            cursor.execute("SELECT * FROM task_scheduled")
            column_names = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            for row in rows:
                row_data = dict(zip(column_names, row))

                cur_b_id = row_data["expr_no"]
                cur_fjspb_index = row_data["fjspb_index"]

                cursor.execute(
                    """
                    SELECT ws_code_fjspb 
                    FROM task_scheduled 
                    WHERE expr_no = ? AND fjspb_index = ?
                    """,
                    (cur_b_id, cur_fjspb_index + 1),
                )
                next_ws_code_fjspb = cursor.fetchone()
                if next_ws_code_fjspb:
                    # Update next_step_ws_code_fjspb for the current row.
                    cursor.execute(
                        """
                        UPDATE task_scheduled 
                        SET next_step_ws_code_fjspb = ? 
                        WHERE expr_no = ? AND fjspb_index = ?""",
                        (next_ws_code_fjspb[0], cur_b_id, cur_fjspb_index),
                    )
            conn.commit()

            # Shift all schedule times left by earliest_time.

            # Find the earliest start time.
            expr_no_placeholders = ",".join("?" * len(expr_no_list))
            cursor.execute(
                f"""
                SELECT MIN(start_time) 
                FROM task_scheduled 
                WHERE expr_no IN ({expr_no_placeholders})
                """,
                expr_no_list,
            )
            earliest_time = cursor.fetchone()[0]

            # Shift in-progress task times left by earliest_time.
            cursor.execute("BEGIN")
            cursor.execute(
                f"SELECT * FROM task_scheduled WHERE expr_no IN ({expr_no_placeholders})",
                expr_no_list,
            )
            column_names = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            for row in rows:
                row_data = dict(zip(column_names, row))
                if row_data["start_time"] and row_data["end_time"]:
                    cursor.execute(
                        """
                            UPDATE task_scheduled 
                            SET start_time = ?, end_time = ? 
                            WHERE expr_no = ? AND fjspb_index = ?""",
                        (
                            row_data["start_time"] - earliest_time,
                            row_data["end_time"] - earliest_time,
                            row_data["expr_no"],
                            row_data["fjspb_index"],
                        ),
                    )
            conn.commit()

            # Update lock_periods for each task.
            cursor.execute("SELECT DISTINCT expr_no FROM task_scheduled")
            all_exprs = [row[0] for row in cursor.fetchall()]

            for expr_no in all_exprs:
                # Read rows in step order.
                cursor.execute(
                    """
                    SELECT fjspb_index, start_time, end_time, lock
                    FROM task_scheduled
                    WHERE expr_no=?
                    ORDER BY fjspb_index
                """,
                    (expr_no,),
                )
                rows = cursor.fetchall()

                lock_periods = []
                in_lock = False
                start_time = None

                for fjspb_index, s_time, e_time, lock in rows:
                    if lock:  # Current step is locked.
                        if not in_lock:
                            # Start a new locked segment.
                            in_lock = True
                            start_time = s_time
                        # Continue the current locked segment.
                    else:
                        if in_lock:
                            # End the locked segment.
                            lock_periods.append([start_time, prev_end_time])
                            in_lock = False
                    prev_end_time = e_time  # Save the previous end_time.

                # Finalize a trailing locked segment.
                if in_lock:
                    lock_periods.append([start_time, prev_end_time])

                # Serialize to JSON and store in the database.
                cursor.execute(
                    """
                    UPDATE task_scheduled
                    SET lock_periods=?
                    WHERE expr_no=?
                """,
                    (json.dumps(lock_periods), expr_no),
                )

            conn.commit()

            # Find makespan.
            expr_no_placeholders = ",".join("?" * len(expr_no_list))
            cursor.execute(
                f"""
                SELECT MAX(end_time) 
                FROM task_scheduled 
                WHERE expr_no IN ({expr_no_placeholders})
                """,
                expr_no_list,
            )
            makespan = cursor.fetchone()[0]

            return makespan

    def fetch_scheduling_table(self, expr_no_list, is_draw=False):
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            expr_no_placeholders = ",".join("?" * len(expr_no_list))
            cursor.execute(
                f"SELECT * FROM task_scheduled WHERE expr_no IN ({expr_no_placeholders}) ORDER BY start_time",
                expr_no_list,
            )

            tasks_info = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            raw_ws_table = defaultdict(list)
            raw_bottle_table = defaultdict(list)

            adjusted_ws_table = defaultdict(list)
            temp_bottle_table = defaultdict(list)

            for row in tasks_info:
                full_task_data = dict(zip(columns, row))
                full_task_data["expr_id"] = self._string_id(full_task_data["expr_id"])
                full_task_data["step_id"] = self._string_id(full_task_data["step_id"])
                if (
                    full_task_data["ws_code_fjspb"] and full_task_data["start_time"] < sys.maxsize
                ) or is_draw:
                    ws_code = full_task_data["ws_code_fjspb"]
                    b_id = full_task_data["expr_id"]
                    expr_no = full_task_data["expr_no"]

                    # Original table format.
                    raw_ws_table[ws_code].append(full_task_data)
                    raw_bottle_table[expr_no].append(full_task_data)

                    # Keep required fields only.
                    adjusted_task_data = {
                        "stepId": full_task_data["step_id"],
                        "startTime": full_task_data["start_time"],
                        "endTime": full_task_data["end_time"],
                        "expr_no": full_task_data["expr_no"],
                        "taskInstanceId": full_task_data["expr_id"],
                    }

                    adjusted_ws_table[ws_code].append(adjusted_task_data)
                    temp_bottle_table[b_id].append(adjusted_task_data)

            adjusted_bottle_table = [
                {
                    "taskInstanceId": task_id,
                    "startTime": min([step["startTime"] for step in steps]),
                    "endTime": max([step["endTime"] for step in steps if step["endTime"]]),
                    "stepList": steps,
                }
                for task_id, steps in temp_bottle_table.items()
            ]

            return raw_ws_table, raw_bottle_table, adjusted_ws_table, adjusted_bottle_table

    # Find a step by primary key (expr_no, fjspb_index).
    def fetch_step_info(self, expr_no, fjspb_index):
        if expr_no is None or fjspb_index is None:
            return None

        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM task_scheduled
                WHERE expr_no = ? AND fjspb_index = ?
                """,
                (
                    expr_no,
                    fjspb_index,
                ),
            )
            column_names = [description[0] for description in cursor.description]
            row = cursor.fetchone()

            if row is None:
                return None

            step_data = dict(zip(column_names, row))
            step_data["expr_id"] = self._string_id(step_data["expr_id"])
            step_data["step_id"] = self._string_id(step_data["step_id"])
            return step_data

    # Find all steps belonging to expr_no.
    def fetch_task_info(self, expr_no):
        if expr_no is None:
            return None

        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            task_data = defaultdict()

            cursor.execute(
                "SELECT * FROM task_info WHERE expr_no = ?",
                (expr_no,),
            )
            column_names = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            if row is None:
                return None

            task_info = dict(zip(column_names, row))

            task_data.update(
                {
                    "task_group": task_info["task_group"],
                    "expr_no": task_info["expr_no"],
                    "expr_name": task_info["expr_name"],
                    "expr_id": self._string_id(task_info["expr_id"]),
                    "status": task_info["status"],
                    "stamp": task_info["stamp"],
                    "abstract_status": task_info["abstract_status"],
                    "is_scheduled": task_info["is_scheduled"],
                }
            )

            # Preserve ordering.
            cursor.execute(
                "SELECT * FROM task_scheduled WHERE expr_no = ? ORDER BY fjspb_index",
                (expr_no,),
            )
            column_names = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            if rows is None:
                return None

            task_data["steps"] = []
            for row in rows:
                step_data = dict(zip(column_names, row))
                step_data["expr_id"] = self._string_id(step_data["expr_id"])
                step_data["step_id"] = self._string_id(step_data["step_id"])
                task_data["steps"].append(step_data)
            return task_data

    # Find the workstation sectionList.
    def fetch_ws_sections(self, ws_code) -> str:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            if ws_code:
                cursor.execute(
                    "SELECT sectionList FROM ws_info WHERE workstationCode = ?",
                    (ws_code,),
                )
                row = cursor.fetchone()
                if row is None:
                    return ""
                return row[0]  # Return sectionList.
            else:
                return ""

    def fetch_ws_by_type_and_version(self, ws_type: str, ws_version: str) -> list:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            if ws_type in ["start", "end"]:
                return ws_type

            cursor.execute(
                """
                SELECT workstationCode 
                FROM ws_info 
                WHERE workstationTypeCode = ? AND version = ?
                """,
                (ws_type, ws_version),
            )
            rows = cursor.fetchall()
            ws_codes = [row[0] for row in rows]
            return ",".join(ws_codes)

    def find_old_task_names(self) -> dict:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            result = defaultdict(dict)

            # Query all tasks.
            cursor.execute("SELECT task_group, expr_no, expr_name FROM task_info")
            tasks = cursor.fetchall()

            for task_group, expr_no, expr_name in tasks:
                # Map a None group to un_scheduled.
                group = task_group if task_group is not None else "un_scheduled"

                # Query only ws_code_fjspb from steps.
                cursor.execute(
                    """SELECT ws_type FROM task_scheduled
                    WHERE expr_no = ?
                    ORDER BY fjspb_index""",
                    (expr_no,),
                )
                steps = [row[0] for row in cursor.fetchall() if row[0] is not None]

                result[group].update({expr_no: {"expr_name": expr_name, "expr_steps": steps}})

            return result

    def update_ptr(self, group, new_ptr):
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            conn.execute("BEGIN")
            cursor.execute("INSERT OR IGNORE INTO ptr_tb (task_group) VALUES (?)", (group,))
            cursor.execute("UPDATE ptr_tb SET ptr = ? WHERE task_group = ?", (new_ptr, group))
            conn.commit()
            return

    def update_makespan(self, group, new_makespan):
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            conn.execute("BEGIN")
            cursor.execute("INSERT OR IGNORE INTO ptr_tb (task_group) VALUES (?)", (group,))
            cursor.execute("UPDATE ptr_tb SET makespan = ? WHERE task_group = ?", (new_makespan, group))
            conn.commit()
            return

    def update_algo_cal_time(self, group, algo_cal_time: float | None):
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            conn.execute("BEGIN")
            cursor.execute("INSERT OR IGNORE INTO ptr_tb (task_group) VALUES (?)", (group,))
            cursor.execute(
                "UPDATE ptr_tb SET algo_cal_time = ? WHERE task_group = ?",
                (algo_cal_time, group),
            )
            conn.commit()
            return

    def find_algo_cal_time(self, task_group: str | None = None) -> float | None:
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            if task_group is None:
                cursor.execute("SELECT MAX(algo_cal_time) FROM ptr_tb")
            else:
                cursor.execute("SELECT algo_cal_time FROM ptr_tb WHERE task_group = ?", (task_group,))
            row = cursor.fetchone()
            if row is None:
                return None

            return row[0]

    # Find the preceding pointer and its actual time for cur_ws_ptr.
    # Find the preceding pointer, then its earliest real_start_time in task_scheduled.
    def find_ws_ptr_old_and_realtime(self, cur_ws_ptr: int):
        """
        Starting from cur_ws_ptr, find the nearest task event time (start or end),
        and return its corresponding real_time.
        """
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()

            # 1. Find the latest event time before cur_ws_ptr.
            cursor.execute(
                """
                SELECT t
                FROM (
                    SELECT start_time AS t
                    FROM task_scheduled
                    WHERE start_time < ?

                    UNION ALL

                    SELECT end_time AS t
                    FROM task_scheduled
                    WHERE end_time < ?
                )
                ORDER BY t DESC
                LIMIT 1;
                """,
                (cur_ws_ptr, cur_ws_ptr),
            )

            row = cursor.fetchone()
            if row is None or row[0] is None:
                return None, None

            ws_ptr_old = row[0]

            # 2. Find its corresponding real_time.
            cursor.execute(
                """
                SELECT
                    CASE
                        WHEN start_time = ? THEN real_start_time
                        WHEN end_time   = ? THEN real_end_time
                    END AS real_t
                FROM task_scheduled
                WHERE start_time = ? OR end_time = ?
                AND real_t IS NOT NULL
                ORDER BY real_t
                LIMIT 1;
                """,
                (ws_ptr_old, ws_ptr_old, ws_ptr_old, ws_ptr_old),
            )

            row = cursor.fetchone()
            ws_ptr2realtime = row[0] if row else None

            return ws_ptr_old, ws_ptr2realtime


def find_idle_periods(conn_sql: sqlite3.Connection, min_gap=80):
    cursor = conn_sql.cursor()
    # Collect all start_time and end_time values.
    cursor.execute("SELECT start_time, end_time FROM task_scheduled")
    rows = cursor.fetchall()

    times = []
    for s, e in rows:
        if s is not None:
            times.append(s)
        if e is not None:
            times.append(e)

    # Sort.
    times = sorted(set(times))

    idle_periods = []
    for i in range(len(times) - 1):
        gap = times[i + 1] - times[i]
        if gap > min_gap:
            idle_periods.append((times[i], times[i + 1]))

    return idle_periods
