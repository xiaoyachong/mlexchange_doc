"""Thin async wrapper around the Prefect API used by the Job API.

The Job API only *creates* and *reads* flow runs; it never runs flows itself,
so it needs the Prefect client but no worker, no PyTorch and no parent flow.
``PREFECT_API_URL`` (and ``PREFECT_API_KEY`` if used) come from the environment.
"""

from typing import Optional

from prefect import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterTags
from prefect.client.schemas.objects import FlowRun, StateType
from prefect.client.schemas.sorting import FlowRunSort
from prefect.states import Cancelling

from flows.chain import JOB_TAG, STEP_TAG, STEPS_TAG, flow_run_parameters, job_tags

TERMINAL = {StateType.COMPLETED, StateType.FAILED, StateType.CANCELLED, StateType.CRASHED}


async def submit_job(job_id: str, steps: list[dict], extra_tags: Optional[list[str]] = None) -> str:
    """Create the flow run of the first step; later steps are chained by the workers."""
    first, rest = steps[0], steps[1:]
    async with get_client() as client:
        deployment = await client.read_deployment_by_name(first["deployment"])
        flow_run = await client.create_flow_run_from_deployment(
            deployment.id,
            parameters=flow_run_parameters(first, "", rest),
            name=first["name"],
            tags=job_tags(job_id, 0, len(steps)) + list(extra_tags or []),
            idempotency_key=f"{job_id}:step0",
        )
    return str(flow_run.id)


async def read_job_runs(job_id: str) -> list[FlowRun]:
    """All flow runs of a job, oldest first."""
    async with get_client() as client:
        return await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(tags=FlowRunFilterTags(all_=[f"{JOB_TAG}{job_id}"])),
            sort=FlowRunSort.EXPECTED_START_TIME_ASC,
            limit=50,
        )


async def cancel_job(job_id: str) -> list[str]:
    """Move every non-terminal flow run of the job to Cancelling."""
    cancelled = []
    async with get_client() as client:
        runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(tags=FlowRunFilterTags(all_=[f"{JOB_TAG}{job_id}"]))
        )
        for run in runs:
            if run.state and run.state.type not in TERMINAL:
                await client.set_flow_run_state(run.id, Cancelling(), force=True)
                cancelled.append(str(run.id))
    return cancelled


async def ping() -> bool:
    """True if the Prefect API answers."""
    async with get_client() as client:
        response = await client.hello()
        return response.status_code == 200


def tag_value(run: FlowRun, prefix: str) -> Optional[int]:
    for tag in run.tags or []:
        if tag.startswith(prefix):
            try:
                return int(tag[len(prefix):])
            except ValueError:
                return None
    return None


def step_index(run: FlowRun) -> Optional[int]:
    return tag_value(run, STEP_TAG)


def total_steps(run: FlowRun) -> Optional[int]:
    return tag_value(run, STEPS_TAG)
