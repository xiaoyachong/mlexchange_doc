from typing import List, Optional

from pydantic import BaseModel, Field


class SlurmParams(BaseModel):
    job_name: str
    num_nodes: int
    partitions: Optional[List[str]] = []
    reservations: Optional[List[str]] = []
    max_time: str = Field(pattern=r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")
    conda_env_name: str
    # TODO: Enforce port pair format
    forward_ports: Optional[List[str]] = []
    submission_ssh_key: Optional[str] = None
    python_file_name: str = Field(
        description="Python file to run", default="src/train.py"
    )
    params: Optional[dict] = {}
