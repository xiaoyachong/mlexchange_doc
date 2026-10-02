"""Request / response models of the Job API."""

from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

TaskName = Literal["train", "inference", "tune", "execute"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "unknown"]


class StepRequest(BaseModel):
    """One step of a job: run one MLflow-registered algorithm once."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(description="Registered model name in MLflow, e.g. 'DLSIA MSDNet'")
    task_name: TaskName = Field(description="Which python_file_* of the algorithm to run")
    model_version: Optional[str] = Field(
        default=None, description="MLflow model version; latest version if omitted"
    )
    resources: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides for slurm/sfapi steps, e.g. {'max_time': '2:00:00'}",
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters (io_parameters, model_parameters, ...)"
    )


class JobRequest(BaseModel):
    """A job is an ordered list of steps; step N+1 starts after step N succeeds.

    All steps of a job run on ONE target, i.e. one child work pool.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    steps: list[StepRequest] = Field(
        min_length=1,
        max_length=10,
        # "params_list" keeps the old launch_parent_flow payload working
        validation_alias=AliasChoices("steps", "params_list"),
    )
    target: Optional[str] = Field(
        default=None,
        description="Compute target for the whole job (e.g. 'als', 'nersc'); config default if omitted",
    )
    job_name: Optional[str] = Field(default=None, max_length=64)


class StepInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    index: int
    model_name: str
    task_name: str
    target: str
    flow_type: str
    deployment: str


class JobSubmitted(BaseModel):
    job_id: str
    first_flow_run_id: str
    steps: list[StepInfo]
    status_url: str


class StepStatus(BaseModel):
    index: int
    flow_run_id: str
    flow_run_name: str
    deployment_id: Optional[str]
    state: str
    state_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_steps: Optional[int]
    steps: list[StepStatus]
    # Run id of the last completed step: this is the uid_save to read results with
    result_uid: Optional[str] = None


class CancelResponse(BaseModel):
    job_id: str
    cancelled_flow_runs: list[str]
