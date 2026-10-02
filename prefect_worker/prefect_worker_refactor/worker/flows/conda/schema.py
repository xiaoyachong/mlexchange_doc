from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CondaParams(BaseModel):
    conda_env_name: str
    python_file_name: str = Field(
        description="Python file to run", default="src/train.py"
    )
    folder_name: str = Field(
        description="Folder name extracted from image_name", default=""
    )
    params: Optional[dict] = {}

    model_config = ConfigDict(extra="forbid")
