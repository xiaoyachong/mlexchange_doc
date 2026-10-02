from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SFAPIParams(BaseModel):
    """
    Parameters for SFAPI (NERSC Superfacility API) job execution.
    """
    job_name: str = Field(description="Name of the SLURM job")
    login_method: Literal["sfapi", "iriapi"] = Field(
        description="NERSC API: 'sfapi' (Iris client key) or 'iriapi' (Globus token)",
        default="sfapi",
    )
    iri_api_base_url: str = Field(
        description="IRI API base URL (login_method=iriapi)",
        default="https://api.iri.nersc.gov",
    )
    iri_job_resource: str = Field(
        description="IRI resource id for Perlmutter job submission",
        default="3cf3c048-855e-4dd8-a189-065a483954bb",
    )
    iri_status_resource: str = Field(
        description="IRI resource id used for job status",
        default="compute",
    )
    iri_login_resource: str = Field(
        description="IRI resource id for Perlmutter login (filesystem ops)",
        default="e525a224-61c1-419f-9642-91168c792e39",
    )
    machine: str = Field(
        description="NERSC machine to run on (currently only 'perlmutter')",
        default="perlmutter"
    )
    queue: str = Field(
        description="SLURM queue/QOS (e.g., 'realtime', 'debug', 'preempt')",
        default="realtime"
    )
    account: str = Field(
        description="NERSC account to charge (e.g., 'als')",
        default="als"
    )
    constraint: str = Field(
        description="Node constraint (e.g., 'cpu', 'gpu')",
        default="cpu"
    )
    num_nodes: int = Field(description="Number of nodes", default=1)
    ntasks_per_node: int = Field(description="Number of tasks per node", default=1)
    cpus_per_task: int = Field(description="CPUs per task", default=64)
    gpus_per_node: int = Field(
        description="GPUs per node (0 = CPU job). >0 adds --gpus-per-node and podman-hpc --gpu",
        default=0,
    )
    max_time: str = Field(
        description="Maximum walltime (HH:MM:SS format)",
        pattern=r"^([0-9]+:)?[0-5]?[0-9]:[0-5][0-9]$",
        default="0:15:00"
    )
    exclusive: bool = Field(
        description="Request exclusive node access",
        default=True
    )
    image_name: str = Field(description="Container image to run")
    image_tag: str = Field(description="Container image tag", default="latest")
    command: str = Field(
        description="Command to run inside the container",
        default="python src/train.py"
    )
    volumes: Optional[List[str]] = Field(
        description="List of volume mounts (host:container format)",
        default=[]
    )
    working_dir: Optional[str] = Field(
        description="Working directory path on NERSC",
        default=""
    )
    output_dir: Optional[str] = Field(
        description="Directory for stdout logs",
        default=""
    )
    error_dir: Optional[str] = Field(
        description="Directory for stderr logs",
        default=""
    )
    params: Optional[dict] = Field(
        description="Job parameters to pass to the script",
        default={}
    )

    model_config = ConfigDict(extra="forbid")
