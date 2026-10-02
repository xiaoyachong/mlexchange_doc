"""Step chaining for multi-step jobs, without a parent worker.

The Job API resolves every step of a job up front (MLflow lookup, target
compute, parameters) and submits only the first step. Each child flow
receives the remaining steps in ``next_steps`` and, when it succeeds,
submits the next one with ``prev_flow_run_id`` set to its own flow run id.

A step is a plain dict, built by ``job_api.routing``::

    {
        "index": 1,                                   # 0-based position in the job
        "deployment": "SFAPI flow/launch_sfapi",       # <flow name>/<deployment name>
        "param_key": "sfapi_params",                   # flow argument holding the params
        "params": {...},                               # validated child params
        "name": "<job_id>-step1-<model>-<task>",       # flow run name
    }

All flow runs of one job share the tags ``mlex-job:<job_id>`` and
``mlex-steps:<n>``; each also has ``mlex-step:<index>``. The Job API uses
these tags to report job status and to cancel jobs.
"""

from typing import Any, Optional

from prefect import get_client
from prefect.context import get_run_context

JOB_TAG = "mlex-job:"
STEP_TAG = "mlex-step:"
STEPS_TAG = "mlex-steps:"


def prepare_io_parameters(params: dict, prev_flow_run_id: str, current_flow_run_id: str) -> dict:
    """Set ``uid_retrieve`` / ``uid_save`` in ``params["io_parameters"]``.

    ``uid_retrieve`` is only filled from the previous step when the caller did
    not set it explicitly (same rule as the original child flows).
    """
    io_params = params.setdefault("io_parameters", {})
    if prev_flow_run_id and not io_params.get("uid_retrieve"):
        io_params["uid_retrieve"] = prev_flow_run_id
    io_params["uid_save"] = current_flow_run_id
    return params


def job_tags(job_id: str, step_index: int, total_steps: int) -> list[str]:
    """Tags attached to every flow run of a job."""
    return [f"{JOB_TAG}{job_id}", f"{STEP_TAG}{step_index}", f"{STEPS_TAG}{total_steps}"]


def flow_run_parameters(step: dict, prev_flow_run_id: str, next_steps: list[dict]) -> dict:
    """Build the Prefect parameters for a step's flow run."""
    return {
        step["param_key"]: step["params"],
        "prev_flow_run_id": prev_flow_run_id,
        "next_steps": next_steps,
    }


def _job_id_from_tags(tags: list[str]) -> Optional[str]:
    for tag in tags:
        if tag.startswith(JOB_TAG):
            return tag[len(JOB_TAG):]
    return None


def submit_next_step(next_steps: Optional[list[dict]], current_flow_run_id: str) -> Optional[str]:
    """Submit the next step of the job, if any. Call only after success.

    Uses an idempotency key, so a retried child flow does not submit the
    next step twice.

    Returns:
        The new flow run id, or None if this was the last step.
    """
    if not next_steps:
        return None

    step, remaining = next_steps[0], next_steps[1:]
    tags = list(get_run_context().flow_run.tags or [])
    job_id = _job_id_from_tags(tags) or current_flow_run_id
    total = step["index"] + 1 + len(remaining)
    # keep other tags of the job (e.g. mlex-client:<name>)
    extra_tags = [t for t in tags if not t.startswith((JOB_TAG, STEP_TAG, STEPS_TAG))]

    with get_client(sync_client=True) as client:
        deployment = client.read_deployment_by_name(step["deployment"])
        flow_run = client.create_flow_run_from_deployment(
            deployment.id,
            parameters=flow_run_parameters(step, current_flow_run_id, remaining),
            name=step.get("name"),
            tags=job_tags(job_id, step["index"], total) + extra_tags,
            idempotency_key=f"{job_id}:step{step['index']}",
        )
    return str(flow_run.id)


def chain_or_return(result: Any, next_steps: Optional[list[dict]], current_flow_run_id: str) -> Any:
    """Return ``result``; if it is the success value (the run id), chain first.

    Child flows return their flow run id on success and a ``Failed`` state on
    failure, so the next step is only submitted on success.
    """
    if result == current_flow_run_id:
        submit_next_step(next_steps, current_flow_run_id)
    return result
