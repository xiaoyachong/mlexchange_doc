"""MLExchange Inference Service: a dedicated endpoint for interactive inference.

This is the only component that imports PyTorch. It runs as its own
container / process (ideally on a GPU host), so neither the application
backend, the Job API nor the Prefect workers carry PyTorch.

Use it for small, latency-sensitive requests (e.g. "segment this slice" from
the annotation app). Large or long inference (whole volumes) should be a
batch job submitted through the Job API instead.

Models are loaded from the MLflow registry as pyfunc models and cached:

    POST /v1/models/{model_name}/predict     run inference
    GET  /v1/models                          models currently loaded
    DELETE /v1/models/{model_name}           unload a model
    GET  /health

Run:  uvicorn inference_service.main:app --host 0.0.0.0 --port 8091 --workers 1
      (one worker per GPU: each worker keeps its own model cache in memory)
"""

import base64
import io
import logging
import os
import secrets
import threading
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("inference_service")

MAX_CACHED_MODELS = int(os.getenv("INFERENCE_MAX_CACHED_MODELS", "2"))
MAX_INPUT_BYTES = int(os.getenv("INFERENCE_MAX_INPUT_BYTES", str(64 * 1024 * 1024)))

app = FastAPI(title="MLExchange Inference Service", version="0.1.0")


# --------------------------------------------------------------------------- #
# Array encoding: small arrays as JSON lists, larger ones as base64 .npy
# --------------------------------------------------------------------------- #
def encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_array(data: str) -> np.ndarray:
    raw = base64.b64decode(data, validate=True)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError(f"Input larger than {MAX_INPUT_BYTES} bytes")
    return np.load(io.BytesIO(raw), allow_pickle=False)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_version: Optional[str] = Field(default=None, description="Registry version; latest if omitted")
    inputs: Optional[list] = Field(default=None, description="Input array as nested JSON lists")
    inputs_npy_b64: Optional[str] = Field(default=None, description="Input array as base64 .npy")
    dtype: str = Field(default="float32", description="dtype for 'inputs'")
    params: dict[str, Any] = Field(default_factory=dict, description="Passed to pyfunc predict(params=...)")

    @model_validator(mode="after")
    def one_input(self):
        if (self.inputs is None) == (self.inputs_npy_b64 is None):
            raise ValueError("Provide exactly one of 'inputs' or 'inputs_npy_b64'")
        return self

    def array(self) -> np.ndarray:
        if self.inputs_npy_b64 is not None:
            return decode_array(self.inputs_npy_b64)
        return np.asarray(self.inputs, dtype=self.dtype)


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    shape: list[int]
    dtype: str
    outputs_npy_b64: str


# --------------------------------------------------------------------------- #
# Model cache (LRU). Loading is serialized; prediction runs in FastAPI's threadpool.
# --------------------------------------------------------------------------- #
class ModelCache:
    def __init__(self, max_models: int):
        self.max_models = max_models
        self._models: "OrderedDict[tuple[str, str], Any]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _latest_version(model_name: str) -> str:
        from mlflow.tracking import MlflowClient

        versions = MlflowClient().search_model_versions(f"name='{model_name}'")
        if not versions:
            raise LookupError(f"Model {model_name!r} not found in MLflow")
        return str(max(int(v.version) for v in versions))

    @staticmethod
    def _load(model_name: str, version: str):
        import mlflow.pyfunc  # imports the model flavor (e.g. torch) lazily

        return mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

    def get(self, model_name: str, version: Optional[str]) -> tuple[str, Any]:
        version = version or self._latest_version(model_name)
        key = (model_name, version)
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return version, self._models[key]
            logger.info(f"Loading model {model_name} v{version}")
            model = self._load(model_name, version)
            self._models[key] = model
            while len(self._models) > self.max_models:
                evicted, _ = self._models.popitem(last=False)
                logger.info(f"Evicted model {evicted}")
            return version, model

    def loaded(self) -> list[dict]:
        with self._lock:
            return [{"model_name": n, "model_version": v} for n, v in self._models]

    def unload(self, model_name: str) -> int:
        with self._lock:
            keys = [k for k in self._models if k[0] == model_name]
            for k in keys:
                del self._models[k]
            return len(keys)


cache = ModelCache(MAX_CACHED_MODELS)


# --------------------------------------------------------------------------- #
# Auth: same X-API-Key scheme as the Job API (INFERENCE_API_KEYS="name:key,...")
# --------------------------------------------------------------------------- #
def require_client(x_api_key: Optional[str] = Header(default=None)) -> str:
    if os.getenv("INFERENCE_ALLOW_ANONYMOUS", "").lower() == "true":
        return "anonymous"
    pairs = [p.partition(":") for p in os.getenv("INFERENCE_API_KEYS", "").split(",") if ":" in p]
    if not pairs:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Inference service has no API keys configured")
    for name, _, key in pairs:
        if x_api_key and key and secrets.compare_digest(x_api_key, key.strip()):
            return name.strip()
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": len(cache.loaded())}


@app.get("/v1/models")
def list_models(client: str = Depends(require_client)):
    return cache.loaded()


@app.delete("/v1/models/{model_name}")
def unload_model(model_name: str, client: str = Depends(require_client)):
    return {"unloaded": cache.unload(model_name)}


@app.post("/v1/models/{model_name}/predict", response_model=PredictResponse)
def predict(model_name: str, request: PredictRequest, client: str = Depends(require_client)):
    # Sync endpoint on purpose: FastAPI runs it in a threadpool, so model
    # loading / GPU inference never blocks the event loop.
    try:
        inputs = request.array()
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:
        version, model = cache.get(model_name, request.model_version)
    except LookupError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except Exception as e:
        logger.exception("Model load failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not load model: {e}")

    try:
        outputs = model.predict(inputs, params=request.params or None)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Prediction failed: {e}")

    outputs = np.asarray(outputs)
    return PredictResponse(
        model_name=model_name,
        model_version=version,
        shape=list(outputs.shape),
        dtype=str(outputs.dtype),
        outputs_npy_b64=encode_array(outputs),
    )
