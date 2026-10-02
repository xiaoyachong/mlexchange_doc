from unittest.mock import MagicMock, patch

from flows import chain


def test_prepare_io_parameters_keeps_explicit_uid():
    p = {"io_parameters": {"uid_retrieve": "explicit"}}
    chain.prepare_io_parameters(p, "prev", "cur")
    assert p["io_parameters"] == {"uid_retrieve": "explicit", "uid_save": "cur"}
    q = {}
    chain.prepare_io_parameters(q, "prev", "cur")
    assert q["io_parameters"] == {"uid_retrieve": "prev", "uid_save": "cur"}


def _steps():
    return [
        {"index": 1, "deployment": "SFAPI flow/launch_sfapi", "param_key": "sfapi_params",
         "params": {"a": 1}, "name": "j-step1"},
        {"index": 2, "deployment": "Docker flow/launch_docker", "param_key": "docker_params",
         "params": {"b": 2}, "name": "j-step2"},
    ]


def test_submit_next_step_passes_rest_and_tags():
    client = MagicMock()
    client.__enter__.return_value = client
    client.create_flow_run_from_deployment.return_value.id = "run-2"
    ctx = MagicMock()
    ctx.flow_run.tags = ["mlex-job:abc", "mlex-step:0", "mlex-steps:3", "mlex-client:annotation-app"]
    with patch.object(chain, "get_client", return_value=client), \
         patch.object(chain, "get_run_context", return_value=ctx):
        assert chain.submit_next_step(_steps(), "run-1") == "run-2"
    client.read_deployment_by_name.assert_called_once_with("SFAPI flow/launch_sfapi")
    kw = client.create_flow_run_from_deployment.call_args.kwargs
    assert kw["parameters"]["sfapi_params"] == {"a": 1}
    assert kw["parameters"]["prev_flow_run_id"] == "run-1"
    assert [s["index"] for s in kw["parameters"]["next_steps"]] == [2]
    assert kw["tags"] == ["mlex-job:abc", "mlex-step:1", "mlex-steps:3", "mlex-client:annotation-app"]
    assert kw["idempotency_key"] == "abc:step1"


def test_chain_only_on_success():
    with patch.object(chain, "submit_next_step") as submit:
        failed = MagicMock()
        assert chain.chain_or_return(failed, _steps(), "run-1") is failed
        submit.assert_not_called()
        assert chain.chain_or_return("run-1", _steps(), "run-1") == "run-1"
        submit.assert_called_once()


def test_last_step_submits_nothing():
    assert chain.submit_next_step([], "run-1") is None
