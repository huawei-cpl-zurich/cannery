"""Refuse supervisor/toolchain upgrades while stopped writers retain active work."""

import json

from cannery.maintenance.capture import Docker, inventory

STAGES = {
    "registrations": ("queued", "running"),
    "submission_intake": ("queued", "intaking"),
    "submission_builds": ("queued", "building"),
    "numerical_baselines": ("queued", "generating"),
    "submission_numerics": ("queued", "validating"),
    "performance_jobs": ("queued", "running"),
    "publication_jobs": ("queued", "running"),
}


def require_drained(docker: Docker, name: str) -> dict[str, int]:
    project = inventory(docker, name)  # All application writers must already be stopped.
    rows = " UNION ALL ".join(
        f"SELECT '{table}' AS stage,count(*) AS active FROM {table} "
        f"WHERE status IN ('{states[0]}','{states[1]}')"
        for table, states in STAGES.items()
    )
    counts: dict[str, int] = json.loads(
        project.sql(f"SELECT json_object_agg(stage,active) FROM ({rows}) t")
    )
    if (
        not isinstance(counts, dict)
        or counts.keys() != STAGES.keys()
        or any(type(v) is not int or v < 0 for v in counts.values())
    ):
        raise ValueError("Invalid drain inventory")
    if any(counts.values()):
        pending = ", ".join(f"{table}={count}" for table, count in counts.items() if count)
        raise ValueError("Restart the OLD deployment to drain retained work: " + pending)
    if inventory(docker, name) != project:
        raise ValueError("Project changed during drain check")
    return counts
