import os


# Resolve full or a positive integer to the available CP-SAT worker count.
def resolve_cp_cpu_threads(cpu_threads="full") -> int:
    if cpu_threads is None:
        cpu_threads = "full"

    if isinstance(cpu_threads, str):
        normalized = cpu_threads.strip().lower()
        if normalized == "full":
            try:
                available_cpus = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                available_cpus = os.cpu_count() or 1
            return max(1, int(available_cpus))
        try:
            cpu_threads = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"threads must be 'full' or a positive integer, got {cpu_threads!r}"
            ) from exc

    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads <= 0:
        raise ValueError(
            f"threads must be 'full' or a positive integer, got {cpu_threads!r}"
        )
    return cpu_threads


# Configure workers through one entry point for consistent OR-Tools settings across solve stages.
def configure_cp_solver_workers(solver, cpu_threads="full") -> int:
    resolved_threads = resolve_cp_cpu_threads(cpu_threads)
    solver.parameters.num_search_workers = resolved_threads
    return resolved_threads
