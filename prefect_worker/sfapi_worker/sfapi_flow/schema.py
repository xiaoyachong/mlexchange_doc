from typing import Optional, List

from pydantic import BaseModel, Field


class SFAPIParams(BaseModel):
    """
    Parameters for SFAPI (NERSC Superfacility API) job execution.
    """
    job_name: str = Field(description="Name of the SLURM job")
    machine: str = Field(
        description="NERSC machine to run on (e.g., 'perlmutter')",
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
    
    class Config:
        extra = "forbid"
