from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from conftest import CONFIG, fake_details
from job_api import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JOB_API_KEYS", "annotation-app:secret-1")
    monkeypatch.setenv("JOB_API_CONFIG", CONFIG)
    monkeypatch.setenv("TILED_STORAGE_DIR", "/data/tiled")
    with patch("job_api.routing.get_algorithm_details", side_effect=fake_details):
        yield TestClient(main.app)


H = {"X-API-Key": "secret-1"}
FINETUNE = {
    "target": "nersc",
    "steps": [
        {"model_name": "DLSIA MSDNet", "task_name": "train", "resources": {"max_time": "1:00:00"},
         "params": {"io_parameters": {"data_tiled_uri": "http://tiled/api/v1/x"}}},
        {"model_name": "DLSIA MSDNet", "task_name": "inference", "params": {}},
    ],
}


def run(state, step, total, rid):
    return SimpleNamespace(
        id=rid, name=f"r{step}", deployment_id=None, start_time=None, end_time=None,
        tags=["mlex-job:x", f"mlex-step:{step}", f"mlex-steps:{total}"],
        state=SimpleNamespace(type=SimpleNamespace(value=state), message=None),
    )


def test_auth_required(client):
    assert client.post("/api/v1/jobs", json=FINETUNE).status_code == 401
    assert client.post("/api/v1/jobs", json=FINETUNE, headers={"X-API-Key": "nope"}).status_code == 401


def test_no_keys_configured_is_503(client, monkeypatch):
    monkeypatch.delenv("JOB_API_KEYS")
    assert client.get("/api/v1/jobs/x", headers=H).status_code == 503


def test_submit_finetune_then_inference_same_pool(client):
    with patch.object(main.gw, "submit_job", AsyncMock(return_value="run-0")) as submit:
        r = client.post("/api/v1/jobs", json=FINETUNE, headers=H)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["first_flow_run_id"] == "run-0"
    # one job -> one target -> one child work pool
    assert [(s["target"], s["deployment"]) for s in body["steps"]] == [("nersc", "SFAPI flow/launch_sfapi")] * 2
    job_id, steps = submit.call_args.args
    assert steps[0]["params"]["max_time"] == "1:00:00"
    assert submit.call_args.kwargs["extra_tags"] == ["mlex-client:annotation-app"]
    assert body["status_url"].endswith(f"/api/v1/jobs/{job_id}")


def test_bad_request_is_422_and_nothing_submitted(client):
    bad = {"target": "nsls-ii", "steps": [{"model_name": "DLSIA MSDNet", "task_name": "train"}]}
    with patch.object(main.gw, "submit_job", AsyncMock()) as submit:
        r = client.post("/api/v1/jobs", json=bad, headers=H)
    assert r.status_code == 422 and "not allowed" in r.json()["detail"]
    submit.assert_not_called()


def test_per_step_target_rejected(client):
    mixed = {"target": "nersc", "steps": [{"model_name": "DLSIA MSDNet", "task_name": "train", "target": "als"}]}
    with patch.object(main.gw, "submit_job", AsyncMock()) as submit:
        assert client.post("/api/v1/jobs", json=mixed, headers=H).status_code == 422
    submit.assert_not_called()


def test_unknown_model_is_404(client):
    from job_api.mlflow_lookup import ModelNotFound
    with patch("job_api.routing.get_algorithm_details", side_effect=ModelNotFound("nope")):
        r = client.post("/api/v1/jobs", json=FINETUNE, headers=H)
    assert r.status_code == 404


@pytest.mark.parametrize("states,expected,result", [
    ([("COMPLETED", 0)], "running", "a0"),                         # step 2 not submitted yet
    ([("COMPLETED", 0), ("RUNNING", 1)], "running", "a0"),
    ([("COMPLETED", 0), ("COMPLETED", 1)], "completed", "a1"),
    ([("COMPLETED", 0), ("FAILED", 1)], "failed", "a0"),
    ([("SCHEDULED", 0)], "queued", None),
    ([("CANCELLING", 0)], "cancelled", None),
])
def test_status(client, states, expected, result):
    runs = [run(s, i, 2, f"a{i}") for s, i in states]
    with patch.object(main.gw, "read_job_runs", AsyncMock(return_value=runs)):
        body = client.get("/api/v1/jobs/x", headers=H).json()
    assert body["status"] == expected and body["total_steps"] == 2
    assert body["result_uid"] == result


def test_status_unknown_job_404(client):
    with patch.object(main.gw, "read_job_runs", AsyncMock(return_value=[])):
        assert client.get("/api/v1/jobs/missing", headers=H).status_code == 404


def test_cancel(client):
    with patch.object(main.gw, "read_job_runs", AsyncMock(return_value=[run("RUNNING", 0, 1, "a0")])), \
         patch.object(main.gw, "cancel_job", AsyncMock(return_value=["a0"])):
        r = client.delete("/api/v1/jobs/x", headers=H)
    assert r.json() == {"job_id": "x", "cancelled_flow_runs": ["a0"]}
