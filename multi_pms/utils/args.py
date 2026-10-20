import argparse


# Parse service arguments separately from RL arguments.
def build_argument_parser():
    parser = argparse.ArgumentParser(description="Run scheduling service with specified runtime options.")

    parser.add_argument("--debug", action="store_true", default=False, help="enable debug mode")
    parser.add_argument("--recover", action="store_true", default=False, help="enable recover mode")
    # Allow CLI overrides of runtime JSON settings that could conflict between concurrent services.
    parser.add_argument(
        "--database_file",
        default="task",
        help="Database sqlite/json basename for scheduling runtime files (default: task).",
    )
    parser.add_argument(
        "--debug_flask_route",
        default="/scheduling_test",
        help="Flask route of the scheduling service in debug mode (default: /scheduling_test).",
    )
    parser.add_argument(
        "--debug_flask_port",
        "--debug_port",
        dest="debug_flask_port",
        default=5081,
        type=int,
        help="Flask port of the scheduling service in debug mode (default: 5081). --debug_port is kept as an alias.",
    )
    parser.add_argument(
        "--schedule_method",
        default="cp_or",
        help="Scheduling method to use (default: cp_or). Common options: cp_or, cp_green_or, cp_green_limit_or, cp_green_limit_strict_or, human, rl_or, rl_release_or, pure_rl.",  # List common values in help without enforcing argparse choices.
    )
    # Expose solver time limits through CLI arguments for independent experiment control.
    parser.add_argument(
        "--schedule_time_limit",
        type=float,
        default=10,
        help="CP/RL-guided scheduling solver time limit in seconds (default: 10).",
    )
    # Control parallel workers consistently across OR-Tools CP-SAT paths.
    parser.add_argument(
        "--threads",
        default="full",
        help="CP-SAT search worker count: 'full' or a positive integer (default: full).",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        default=False,
        help="fix one process-wide peak/valley interval from service startup time; --recover restores it",
    )

    return parser


# allow_unknown ignores the other parser's arguments when service and RL share a command line.
def parse_arguments(argv=None, allow_unknown=False):
    parser = build_argument_parser()
    if allow_unknown:
        args, _ = parser.parse_known_args(argv)
    else:
        args = parser.parse_args(argv)
    return args
