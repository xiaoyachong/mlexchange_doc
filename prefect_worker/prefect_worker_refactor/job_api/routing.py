"""Turn a JobRequest into concrete child-flow steps.

This is the logic that used to live in ``flows/parent_flow.py``. It now runs
inside the Job API request, before anything is submitted, so a bad model
name or missing conda env is a 4xx response instead of a failed parent run.

Each step is validated with the *same* schema classes the child flows use
(imported from ``worker/flows``), so the API and the workers cannot drift.
"""

import os
import re
from typing import Callable, Optional

import yaml

from flows.conda.schema import CondaParams
from flows.docker.schema import DockerParams
from flows.podman.schema import PodmanParams
from flows.sfapi.schema import SFAPIParams
from flows.slurm.schema import SlurmParams
from job_api.mlflow_lookup import get_algorithm_details, python_file_for_task
from job_api.models import JobRequest, StepRequest

# flow type -> (deployment "<flow name>/<deployment name>", flow argument name)
DEPLOYMENTS = {
    "conda": ("launch_conda/launch_conda", "conda_params"),
    "docker": ("Docker flow/launch_docker", "docker_params"),
    "podman": ("Podman flow/launch_podman", "podman_params"),
    "slurm": ("launch_slurm/launch_slurm", "slurm_params"),
    "sfapi": ("SFAPI flow/launch_sfapi", "sfapi_params"),
}


class RoutingError(ValueError):
    """The request cannot be turned into runnable steps (client error)."""


def expand_env_vars(obj):
    """Recursively expand ${VAR} / $VAR in strings."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(path: Optional[str] = None) -> dict:
    """Load the Job API config (``JOB_API_CONFIG`` or job_api/config.yml)."""
    path = path or os.getenv("JOB_API_CONFIG") or os.path.join(os.path.dirname(__file__), "config.yml")
    with open(path, "r") as f:
        return expand_env_vars(yaml.safe_load(f) or {})


def slurm_job_name(*parts: str) -> str:
    """Slurm-safe job name (no spaces / shell characters, <= 50 chars)."""
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", "_".join(p for p in parts if p))[:50]


def _conda_env(config: dict, details: dict) -> str:
    envs = config.get("conda", {}).get("conda_env_name", {})
    name, version = details["model_name"], details.get("algorithm_version", "")
    if version and f"{name}_{version}" in envs:
        return envs[f"{name}_{version}"]
    return envs.get(name, "")


def _as_list(value) -> list:
    if isinstance(value, str):
        return yaml.safe_load(value) or []
    return list(value or [])


def _overrides(config: dict, step: StepRequest) -> dict:
    allowed = set(config.get("job_api", {}).get("allowed_resource_overrides", []))
    unknown = set(step.resources) - allowed
    if unknown:
        raise RoutingError(f"Resource overrides not allowed: {sorted(unknown)} (allowed: {sorted(allowed)})")
    return dict(step.resources)


# --------------------------------------------------------------------------- #
# Per flow type builders: (step, details, python_file, config) -> params model
# --------------------------------------------------------------------------- #
def _build_conda(step, details, python_file, config):
    env = _conda_env(config, details)
    if not env:
        raise RoutingError(f"No conda env configured for model {details['model_name']!r} (conda.conda_env_name)")
    return CondaParams(
        conda_env_name=env,
        python_file_name=python_file,
        folder_name=details["image_name"].split("/")[-1],
        params=step.params,
    )


def _build_container(cls):
    def build(step, details, python_file, config):
        container = config.get("container", {})
        return cls(
            image_name=details["image_name"],
            image_tag=details["image_tag"],
            command=f"python {python_file}",
            volumes=container.get("volumes", []),
            network=container.get("network", ""),
            env_vars={},
            params=step.params,
        )
    return build


def _build_slurm(step, details, python_file, config):
    slurm = {**config.get("slurm", {}), **_overrides(config, step)}
    env = _conda_env(config, details)
    if not env:
        raise RoutingError(f"No conda env configured for model {details['model_name']!r} (slurm uses conda)")
    return SlurmParams(
        job_name=slurm_job_name(details["model_name"], step.task_name),
        num_nodes=slurm.get("num_nodes", 1),
        partitions=_as_list(slurm.get("partitions")),
        reservations=_as_list(slurm.get("reservations")),
        max_time=slurm.get("max_time", "1:00:00"),
        conda_env_name=env,
        forward_ports=_as_list(slurm.get("forward_ports")),
        submission_ssh_key=slurm.get("submission_ssh_key", ""),
        python_file_name=python_file,
        params=step.params,
    )


def _build_sfapi(step, details, python_file, config):
    sf = {**config.get("sfapi", {}), **_overrides(config, step)}
    is_gpu = details.get("is_gpu_enabled", False)
    iri = {f"iri_{k}": v for k, v in (sf.get("iri") or {}).items() if v}
    return SFAPIParams(
        job_name=slurm_job_name(details["model_name"], step.task_name, details["image_name"].split("/")[-1]),
        login_method=sf.get("login_method", "sfapi"),
        **iri,
        machine=sf.get("machine", "perlmutter"),
        queue=sf.get("queue", "realtime"),
        account=sf.get("account", "als"),
        constraint="gpu" if is_gpu else sf.get("constraint", "cpu"),
        num_nodes=sf.get("num_nodes", 1),
        ntasks_per_node=sf.get("ntasks_per_node", 1),
        cpus_per_task=sf.get("cpus_per_task", 64),
        gpus_per_node=sf.get("gpus_per_node", 4) if is_gpu else 0,
        max_time=sf.get("max_time", "0:15:00"),
        exclusive=sf.get("exclusive", True),
        image_name=details["image_name"],
        image_tag=details["image_tag"],
        command=f"python {python_file}",
        volumes=sf.get("volumes") or [],
        working_dir=sf.get("working_dir", ""),
        output_dir=sf.get("output_dir", ""),
        error_dir=sf.get("error_dir", ""),
        params=step.params,
    )


BUILDERS = {
    "conda": _build_conda,
    "docker": _build_container(DockerParams),
    "podman": _build_container(PodmanParams),
    "slurm": _build_slurm,
    "sfapi": _build_sfapi,
}


def resolve_target(config: dict, job: JobRequest) -> tuple[str, str]:
    """Return (target name, flow type) for the job, enforcing allowed_targets.

    A job runs on exactly one target, so every step goes to the same child work pool.
    """
    api_cfg = config.get("job_api", {})
    target = (job.target or api_cfg.get("default_target", "als")).lower()
    allowed = [t.lower() for t in api_cfg.get("allowed_targets", [])]
    if allowed and target not in allowed:
        raise RoutingError(f"Target {target!r} is not allowed (allowed: {allowed})")
    flow_type = config.get("targets", {}).get(target)
    if flow_type not in BUILDERS:
        raise RoutingError(f"Target {target!r} is not mapped to a known flow type")
    return target, flow_type


def build_steps(
    job: JobRequest,
    job_id: str,
    config: dict,
    lookup: Optional[Callable[[str, Optional[str]], dict]] = None,
) -> list[dict]:
    """Resolve every step of ``job`` into a chainable step dict (see worker/flows/chain.py).

    ``lookup`` defaults to the MLflow lookup; tests pass a fake.
    """
    lookup = lookup or get_algorithm_details
    steps = []
    target, flow_type = resolve_target(config, job)  # one child work pool per job
    for index, step in enumerate(job.steps):
        details = lookup(step.model_name, step.model_version)

        python_file = python_file_for_task(details, step.task_name)
        if not python_file:
            raise RoutingError(f"Model {step.model_name!r} has no python file for task {step.task_name!r}")
        if flow_type != "conda" and not details.get("image_name"):
            raise RoutingError(f"Model {step.model_name!r} has no image_name in MLflow")

        try:
            params_model = BUILDERS[flow_type](step, details, python_file, config)
        except RoutingError:
            raise
        except Exception as e:  # pydantic validation of the child schema
            raise RoutingError(f"Step {index} ({step.model_name}/{step.task_name}): {e}")

        deployment, param_key = DEPLOYMENTS[flow_type]
        prefix = job.job_name or job_id
        steps.append({
            "index": index,
            "deployment": deployment,
            "param_key": param_key,
            "params": params_model.model_dump(),
            "name": slurm_job_name(prefix, f"step{index}", step.model_name, step.task_name),
            # metadata for the API response only
            "model_name": step.model_name,
            "task_name": step.task_name,
            "target": target,
            "flow_type": flow_type,
        })
    return steps
