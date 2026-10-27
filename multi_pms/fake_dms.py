import argparse
import json
import time
from pathlib import Path

import requests


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _same_identifier(left, right):
    """Compare IDs that may be represented by different JSON types in the DMS and scheduling service."""
    return left is not None and right is not None and str(left) == str(right)


def _assignment_matches_task(assignment, task):
    """Scope scheduling instructions to a task by its task number or task instance ID."""
    expr_no = assignment.get("expr_no")
    task_instance_id = assignment.get("taskInstanceId")

    # Some DMS templates reuse taskInstanceId and stepId across tasks.
    # When the service provides expr_no, prefer it as task identity over duplicated instance IDs.
    if expr_no is not None:
        return _same_identifier(expr_no, task.get("expr_no"))
    return _same_identifier(task_instance_id, task.get("id"))


def mark_completed_steps(payload, response_payload):
    """Write step instructions returned by the scheduling service back to the fake DMS cache payload.

    Returns the list of ``(expr_no, step_id)`` pairs first changed to ``completed`` in this round;
    repeated instructions are not counted again, keeping the write-back idempotent.
    """
    if not isinstance(payload, dict) or not isinstance(response_payload, dict):
        return []

    assignments = response_payload.get("data")
    task_list = payload.get("taskList")
    if not isinstance(assignments, list) or not isinstance(task_list, list):
        return []

    completed_steps = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue

        step_id = assignment.get("stepId", assignment.get("step_id"))
        if step_id is None:
            # Non-step commands such as scheduling_table do not modify task steps.
            continue

        for task in task_list:
            if not isinstance(task, dict) or not _assignment_matches_task(assignment, task):
                continue

            steps = task.get("steps")
            if not isinstance(steps, list):
                break

            for step in steps:
                if not isinstance(step, dict):
                    continue

                cached_step_id = step.get("id", step.get("stepId", step.get("step_id")))
                if (
                    step.get("workstationType") in {"start", "lock", "unlock"}
                    or not _same_identifier(cached_step_id, step_id)
                ):
                    continue

                if step.get("status") != "completed":
                    step["status"] = "completed"
                    completed_steps.append((task.get("expr_no"), step_id))
                break
            break

    return completed_steps


def build_parser():
    parser = argparse.ArgumentParser(description="Fake DMS")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="enable interactive debug mode with a cached and stateful JSON payload",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5081,
        help="scheduling service port (default: 5081)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    port = args.port

    json_file_path = Path(__file__).resolve().parent / "dms_link" / "expr_json" / "1-pbas-1.json"

    # Allow CLI service-port overrides for multiple instances and custom debug ports.
    post_url = f"http://127.0.0.1:{port}/scheduling_test"
    cached_payload = load_json_file(json_file_path) if args.debug else None

    while True:
        try:
            payload = cached_payload if args.debug else load_json_file(json_file_path)
            response = requests.post(post_url, json=payload)
            print(f"\nReceive Response: {response}, {response.text}")

            if args.debug and response.ok:
                try:
                    response_payload = response.json()
                except ValueError:
                    response_payload = None

                completed_steps = mark_completed_steps(payload, response_payload)
                if completed_steps:
                    print(f"Debug fake DMS completed operations: {completed_steps}")
        except Exception as e:
            print("Error:", e)

        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
