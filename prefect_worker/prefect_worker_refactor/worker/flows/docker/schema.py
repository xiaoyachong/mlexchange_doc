from typing import Optional

from pydantic import BaseModel, Field


class DockerParams(BaseModel):
    image_name: str
    image_tag: str
    command: str = Field(
        description="Command to run in docker", default="python src/train.py"
    )
    params: Optional[dict] = {}
    volumes: Optional[list] = []
    network: Optional[str] = ""
    env_vars: Optional[dict] = {}
