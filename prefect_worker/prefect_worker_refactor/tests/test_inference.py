from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference_service import main as inf

H = {"X-API-Key": "k1"}


class Doubler:
    def predict(self, x, params=None):
        return x * 2


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEYS", "annotation-app:k1")
    inf.cache._models.clear()
    loads = []

    def load(name, version):
        loads.append((name, version))
        return Doubler()
    with patch.object(inf.ModelCache, "_load", staticmethod(load)), \
         patch.object(inf.ModelCache, "_latest_version", staticmethod(lambda n: "7")):
        yield TestClient(inf.app), loads


def test_predict_json_and_cache(client):
    c, loads = client
    for _ in range(2):
        r = c.post("/v1/models/DLSIA MSDNet/predict", json={"inputs": [[1, 2], [3, 4]]}, headers=H)
        assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == "7" and body["shape"] == [2, 2]
    np.testing.assert_array_equal(inf.decode_array(body["outputs_npy_b64"]), [[2, 4], [6, 8]])
    assert loads == [("DLSIA MSDNet", "7")]          # loaded once, then cached


def test_predict_npy_and_lru_eviction(client):
    c, loads = client
    arr = inf.encode_array(np.ones((2, 3), dtype=np.float32))
    for name in ["a", "b", "c"]:
        assert c.post(f"/v1/models/{name}/predict", json={"inputs_npy_b64": arr}, headers=H).status_code == 200
    assert [m["model_name"] for m in c.get("/v1/models", headers=H).json()] == ["b", "c"]


def test_validation_and_auth(client):
    c, _ = client
    assert c.post("/v1/models/a/predict", json={"inputs": [1]}).status_code == 401
    assert c.post("/v1/models/a/predict", json={}, headers=H).status_code == 422
    assert c.post("/v1/models/a/predict", json={"inputs_npy_b64": "not-base64!"}, headers=H).status_code == 422
