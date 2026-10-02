"""Look up algorithm details for a registered model in MLflow.

Only metadata is read (run params / tags). Nothing here imports PyTorch, so
the Job API can depend on ``mlflow-skinny`` instead of full ``mlflow``.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

TASK_TO_PARAM = {
    "train": "python_file_train",
    "inference": "python_file_inference",
    "tune": "python_file_tune",
    "execute": "python_file",
}


class ModelNotFound(LookupError):
    """The model (or version) is not registered in MLflow."""


def _client():
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set for the Job API.")
    # MLflow reads MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD from the environment
    return MlflowClient(tracking_uri=tracking_uri)


def _resolve_version(client, model_name: str, model_version: Optional[str]):
    if model_version:
        try:
            return client.get_model_version(model_name, str(model_version))
        except Exception as e:
            raise ModelNotFound(f"Model {model_name!r} version {model_version} not found: {e}")
    # get_latest_versions() is deprecated (stages); pick the highest version number
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ModelNotFound(f"Model {model_name!r} not found in MLflow")
    return max(versions, key=lambda v: int(v.version))


def get_algorithm_details(model_name: str, model_version: Optional[str] = None) -> dict:
    """Return the algorithm details the router needs.

    Returns:
        Dict with model_name, model_version, algorithm_version, image_name,
        image_tag, source, is_gpu_enabled and python_file* entries.
    """
    client = _client()
    version = _resolve_version(client, model_name, model_version)
    run = client.get_run(version.run_id)
    params = run.data.params
    tags = run.data.tags

    details = {
        "model_name": model_name,
        "model_version": str(version.version),
        "algorithm_version": tags.get("version", ""),
        "image_name": params.get("image_name", ""),
        "image_tag": params.get("image_tag", ""),
        "source": params.get("source", ""),
        "is_gpu_enabled": str(params.get("is_gpu_enabled", "False")).lower() == "true",
    }
    for key in TASK_TO_PARAM.values():
        if key in params:
            details[key] = params[key]

    logger.info(f"Resolved {model_name} v{details['model_version']} -> {details['image_name']}")
    return details


def python_file_for_task(details: dict, task_name: str) -> str:
    """Pick the entry-point file for ``task_name`` (falls back to python_file)."""
    return details.get(TASK_TO_PARAM.get(task_name, "python_file"), "") or details.get("python_file", "")
