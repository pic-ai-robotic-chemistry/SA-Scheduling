import copy
import pytz
from datetime import datetime
from pprint import pprint, pformat

from obj.safe_attr import SafeAttr


class Step:
    FIELDS = [
        "task_group",
        "name",
        "expr_id",
        "step_id",
        "expr_no",
        "expr_name",
        "ws_code_fjspb",
        "start_time",
        "duration",
        "time",
        "scaled_time",  # Scaled time: factor 60 for minutes or 1 for seconds.
        "end_time",
        "real_start_time",
        "real_end_time",
        "fjspb_index",
        "status",
    ]

    def __init__(self, step_data, time_tolerance=600, time_unit="seconds"):
        """
        Args:
            step_data (dict): Step data
            time_tolerance (float): Allowed tolerance for early or late completion
            time_unit (str): Tolerance unit; choose "seconds" or "minutes"
        """
        # Tolerance for early or late completion.
        self.unit_factor = 60 if time_unit in ["minutes", "min"] else 1
        self.earlier_limit = time_tolerance * self.unit_factor
        self.later_limit = time_tolerance * self.unit_factor

        self.earlier_completed = False
        self.later_completed = False

        self.update(step_data)

        self.assignment = {
            "scheduleId": "",
            "scheduleType": "step",
            "taskInstanceId": self.expr_id,
            "stepId": self.step_id,
            "expr_no": self.expr_no,
            "workstationCode": "",
            "mockTime": 1,
            "robotCode": "",
            "execCmdRobotList": [],
        }

    @property
    def completed_status(self):
        self.scaled_time = self.time * self.unit_factor if self.time else None

        timezone = pytz.timezone("Asia/Shanghai")
        result = {
            "status": "not_start",
            "detail": {
                "real_start_time": "",
                "real_end_time": "",
                "cur_time": datetime.now(timezone).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
                "estimate_duration": self.scaled_time,
                "earlier_limit": self.earlier_limit,
                "later_limit": self.later_limit,
            },
        }
        # Check self.duration so the first update does not fail.
        if self.duration and self.real_start_time:
            result["status"] = "in_progress"
            result["detail"]["real_start_time"] = self.real_start_time

            if self.real_end_time:
                result["detail"]["real_end_time"] = self.real_end_time
                seconds_diff = (
                    datetime.strptime(self.real_end_time, "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(self.real_start_time, "%Y-%m-%d %H:%M:%S")
                ).total_seconds()
                if seconds_diff + self.earlier_limit < self.scaled_time:
                    result["status"] = "earlier_completed"
                elif seconds_diff - self.later_limit > self.scaled_time:
                    result["status"] = "later_completed"
                else:
                    result["status"] = "normal_completed"
            else:
                seconds_diff = (
                    datetime.now(timezone).replace(tzinfo=None)
                    - datetime.strptime(self.real_start_time, "%Y-%m-%d %H:%M:%S")
                ).total_seconds()
                if seconds_diff - self.later_limit > self.scaled_time:
                    result["status"] = "later_completed"

        return result

    def update(self, step_data):
        for field in self.FIELDS:
            setattr(self, field, step_data.get(field))

        if self.completed_status["status"] in ["earlier_completed"]:
            self.earlier_completed = True
            self.later_completed = False
        elif self.completed_status["status"] in ["later_completed"]:
            self.earlier_completed = False
            self.later_completed = True
        else:
            self.earlier_completed = False
            self.later_completed = False

    def get_attr(self, attr_name):
        return SafeAttr(getattr(self, attr_name, None))

    def do_start(self, schedule_id, robot_code):
        self.assignment.update(
            {
                "scheduleId": "{:06d}".format(schedule_id),
                "workstationCode": self.ws_code_fjspb,
                "robotCode": robot_code,
            }
        )
        return copy.deepcopy(self.assignment)

    def __str__(self) -> str:
        return f"Step(name={self.name}, fjspb_index={self.fjspb_index}, status={self.status}, completed_status={self.completed_status})"

    def __repr__(self) -> str:
        return self.__str__()


class StepFactory:
    @staticmethod
    def create_step(step_data) -> Step:

        return Step(step_data, time_unit="minutes")
