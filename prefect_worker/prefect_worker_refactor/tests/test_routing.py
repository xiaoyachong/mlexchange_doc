import pytest

from conftest import fake_details
from flows.sfapi.schema import SFAPIParams
from job_api.models import JobRequest
from job_api.routing import RoutingError, build_steps, slurm_job_name


def job(**kw):
    base = {"steps": [{"model_name": "DLSIA MSDNet", "task_name": "train", "params": {"io_parameters": {}}}]}
    base.update(kw)
    return JobRequest.model_validate(base)


def test_old_parent_payload_still_accepted():
    req = JobRequest.model_validate({"params_list": [
        {"model_name": "DLSIA MSDNet", "task_name": "train", "params": {}},
        {"model_name": "DLSIA MSDNet", "task_name": "inference", "params": {}},
    ]})
    assert len(req.steps) == 2


def test_default_target_is_docker(config):
    steps = build_steps(job(), "j1", config, lookup=fake_details)
    s = steps[0]
    assert s["flow_type"] == "docker" and s["deployment"] == "Docker flow/launch_docker"
    assert s["params"]["command"] == "python src/train.py"
    assert s["params"]["volumes"] == ["/data/tiled:/tiled_storage"]


def test_nersc_gpu_model_routes_to_sfapi_gpu(config):
    steps = build_steps(job(target="nersc"), "j1", config,
                        lookup=lambda n, v: fake_details(n, v, gpu=True))
    s = steps[0]
    assert s["deployment"] == "SFAPI flow/launch_sfapi" and s["param_key"] == "sfapi_params"
    p = SFAPIParams(**s["params"])          # round-trips through the worker schema
    assert p.constraint == "gpu" and p.gpus_per_node == 4 and p.login_method == "sfapi"
    assert " " not in p.job_name


def test_resource_override_whitelist(config):
    ok = job(target="nersc")
    ok.steps[0].resources = {"max_time": "2:00:00"}
    assert build_steps(ok, "j1", config, lookup=fake_details)[0]["params"]["max_time"] == "2:00:00"
    bad = job(target="nersc")
    bad.steps[0].resources = {"account": "someone_else"}
    with pytest.raises(RoutingError, match="not allowed"):
        build_steps(bad, "j1", config, lookup=fake_details)


def test_target_not_allowed(config):
    with pytest.raises(RoutingError, match="not allowed"):
        build_steps(job(target="nsls-ii"), "j1", config, lookup=fake_details)


def test_missing_python_file(config):
    req = JobRequest.model_validate({"steps": [{"model_name": "X", "task_name": "tune"}]})
    with pytest.raises(RoutingError, match="no python file"):
        build_steps(req, "j1", config, lookup=fake_details)


def test_conda_env_lookup_by_version(config):
    req = JobRequest.model_validate({"target": "conda", "steps": [{"model_name": "DLSIA MSDNet", "task_name": "train"}]})
    s = build_steps(req, "j1", config, lookup=fake_details)[0]
    assert s["params"]["conda_env_name"] == "dlsia_0.1.0"
    assert s["params"]["folder_name"] == "mlex_dlsia_segmentation_prototype"


def test_job_name_sanitized():
    assert slurm_job_name("DLSIA TUNet3+", "train") == "DLSIA_TUNet3+_train"
