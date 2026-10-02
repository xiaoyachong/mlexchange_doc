"""MLExchange Job API: the FastAPI endpoint that replaces the parent worker.

    Frontend / app backend  --HTTP-->  Job API  --Prefect API-->  child workers
                                         |                        (docker, conda,
                                         +-- MLflow (metadata)     slurm, sfapi, ...)

Endpoints
    POST   /api/v1/jobs              submit a job (one or more chained steps)
    GET    /api/v1/jobs/{job_id}     job status, per step
    DELETE /api/v1/jobs/{job_id}     cancel a job
    GET    /health                   liveness
    GET    /ready                    Prefect API reachable

Run:  uvicorn job_api.main:app --host 0.0.0.0 --port 8090
      (with PYTHONPATH containing the repo root and worker/)
"""

import logging
import os
import secrets
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from job_api import prefect_gateway as gw
from job_api.mlflow_lookup import ModelNotFound
from job_api.models import (
    CancelResponse,
    JobRequest,
    JobStatusResponse,
    JobSubmitted,
    StepInfo,
    StepStatus,
)
from job_api.routing import RoutingError, build_steps, load_config

logger = logging.getLogger("job_api")

app = FastAPI(
    title="MLExchange Job API",
    version="0.1.0",
    description="Submit ML training / inference jobs; Prefect orchestrates them on the selected compute.",
)


# --------------------------------------------------------------------------- #
# Auth: X-API-Key header. JOB_API_KEYS="name1:key1,name2:key2"
# --------------------------------------------------------------------------- #
def _api_keys() -> dict[str, str]:
    keys = {}
    for item in filter(None, (s.strip() for s in os.getenv("JOB_API_KEYS", "").split(","))):
        name, _, key = item.partition(":")
        if key:
            keys[key] = name
    return keys


def require_client(x_api_key: Optional[str] = Header(default=None)) -> str:
    """Return the client name for the API key, or raise 401/503."""
    if os.getenv("JOB_API_ALLOW_ANONYMOUS", "").lower() == "true":
        return "anonymous"
    keys = _api_keys()
    if not keys:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job API has no API keys configured")
    for key, name in keys.items():
        if x_api_key and secrets.compare_digest(x_api_key, key):
            return name
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


# --------------------------------------------------------------------------- #
# Status helpers
# --------------------------------------------------------------------------- #
def _overall_status(runs, total: Optional[int]) -> str:
    if not runs:
        return "unknown"
    types = [r.state.type.value if r.state else "UNKNOWN" for r in runs]
    if any(t in ("FAILED", "CRASHED") for t in types):
        return "failed"
    if any(t in ("CANCELLED", "CANCELLING") for t in types):
        return "cancelled"
    if total is not None and types.count("COMPLETED") >= total:
        return "completed"
    if all(t in ("SCHEDULED", "PENDING", "PAUSED") for t in types):
        return "queued"
    return "running"  # a step is running, or the next step is being submitted


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        ok = await gw.ping()
    except Exception as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Prefect API unreachable: {e}")
    return {"status": "ok" if ok else "degraded"}


@app.post("/api/v1/jobs", response_model=JobSubmitted, status_code=status.HTTP_201_CREATED)
async def submit_job(job: JobRequest, request: Request, client_name: str = Depends(require_client)):
    """Validate and resolve all steps, then start the first one."""
    job_id = uuid.uuid4().hex[:12]
    try:
        # MLflow lookups are blocking HTTP calls: keep them off the event loop
        steps = await run_in_threadpool(build_steps, job, job_id, load_config())
    except ModelNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except RoutingError as e:
        raise HTTPException(422, str(e))

    try:
        first_run = await gw.submit_job(job_id, steps, extra_tags=[f"mlex-client:{client_name}"])
    except Exception as e:
        logger.exception("Prefect submission failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not submit to Prefect: {e}")

    logger.info(f"job {job_id} by {client_name}: {len(steps)} step(s), first run {first_run}")
    return JobSubmitted(
        job_id=job_id,
        first_flow_run_id=first_run,
        steps=[StepInfo(**{k: s[k] for k in StepInfo.model_fields}) for s in steps],
        status_url=str(request.url_for("get_job", job_id=job_id)),
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, name="get_job")
async def get_job(job_id: str, client_name: str = Depends(require_client)):
    runs = await gw.read_job_runs(job_id)
    if not runs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")

    total = next((gw.total_steps(r) for r in runs if gw.total_steps(r) is not None), None)
    steps = sorted(
        (
            StepStatus(
                index=gw.step_index(r) if gw.step_index(r) is not None else -1,
                flow_run_id=str(r.id),
                flow_run_name=r.name,
                deployment_id=str(r.deployment_id) if r.deployment_id else None,
                state=r.state.type.value if r.state else "UNKNOWN",
                state_message=r.state.message if r.state else None,
                start_time=_iso(r.start_time),
                end_time=_iso(r.end_time),
            )
            for r in runs
        ),
        key=lambda s: s.index,
    )
    completed = [s for s in steps if s.state == "COMPLETED"]
    return JobStatusResponse(
        job_id=job_id,
        status=_overall_status(runs, total),
        total_steps=total,
        steps=steps,
        result_uid=completed[-1].flow_run_id if completed else None,
    )


@app.delete("/api/v1/jobs/{job_id}", response_model=CancelResponse)
async def cancel_job(job_id: str, client_name: str = Depends(require_client)):
    if not await gw.read_job_runs(job_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    return CancelResponse(job_id=job_id, cancelled_flow_runs=await gw.cancel_job(job_id))
