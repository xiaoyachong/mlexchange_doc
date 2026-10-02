"""Example: what the annotation app backend does instead of calling the parent flow.

1. Submit a fine-tuning job on NERSC (train, then inference; both steps run on the
   sfapi work pool - a job always uses exactly one child work pool).
2. Poll the job until it finishes.
3. Run interactive inference on one slice through the inference service.

    JOB_API_URL=http://localhost:8090 JOB_API_KEY=... \\
    INFERENCE_URL=http://localhost:8091 INFERENCE_API_KEY=... \\
    python examples/annotation_app_client.py

Only needs httpx + numpy: no Prefect, no MLflow, no PyTorch in the app backend.
"""

import base64
import io
import os
import time

import httpx
import numpy as np

JOB_API = os.getenv("JOB_API_URL", "http://localhost:8090")
INFERENCE = os.getenv("INFERENCE_URL", "http://localhost:8091")
JOB_HEADERS = {"X-API-Key": os.environ.get("JOB_API_KEY", "")}
INF_HEADERS = {"X-API-Key": os.environ.get("INFERENCE_API_KEY", "")}

MODEL = "DLSIA MSDNet"


def submit_finetune() -> str:
    job = {
        "job_name": "annotation-finetune",
        "target": "nersc",  # the whole job runs on the sfapi work pool
        "steps": [
            {
                "model_name": MODEL,
                "task_name": "train",
                "resources": {"max_time": "1:00:00"},
                "params": {
                    "io_parameters": {
                        "data_tiled_uri": "https://tiled.example/api/v1/metadata/project/raw",
                        "mask_tiled_uri": "https://tiled.example/api/v1/metadata/project/masks",
                        "uid_retrieve": "",
                    },
                    "model_parameters": {"num_epochs": 10},
                },
            },
            {
                # runs after training succeeds, also on NERSC; reads the trained model via uid_retrieve
                "model_name": MODEL,
                "task_name": "inference",
                "params": {"io_parameters": {
                    "data_tiled_uri": "https://tiled.example/api/v1/metadata/project/raw",
                    "seg_tiled_uri": "https://tiled.example/api/v1/metadata/project/seg",
                }},
            },
        ],
    }
    r = httpx.post(f"{JOB_API}/api/v1/jobs", json=job, headers=JOB_HEADERS, timeout=30)
    r.raise_for_status()
    body = r.json()
    print(f"job {body['job_id']}: {[(s['task_name'], s['target']) for s in body['steps']]}")
    return body["job_id"]


def wait(job_id: str, poll_seconds: int = 30) -> dict:
    while True:
        r = httpx.get(f"{JOB_API}/api/v1/jobs/{job_id}", headers=JOB_HEADERS, timeout=30)
        r.raise_for_status()
        status = r.json()
        print(f"  {status['status']}: {[(s['index'], s['state']) for s in status['steps']]}")
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(poll_seconds)


def interactive_inference(slice_2d: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    np.save(buffer, slice_2d.astype(np.float32), allow_pickle=False)
    r = httpx.post(
        f"{INFERENCE}/v1/models/{MODEL}/predict",
        json={"inputs_npy_b64": base64.b64encode(buffer.getvalue()).decode()},
        headers=INF_HEADERS,
        timeout=120,
    )
    r.raise_for_status()
    return np.load(io.BytesIO(base64.b64decode(r.json()["outputs_npy_b64"])), allow_pickle=False)


if __name__ == "__main__":
    final = wait(submit_finetune())
    print("result uid (uid_save of last step):", final["result_uid"])
    print("prediction shape:", interactive_inference(np.random.rand(1, 1, 256, 256)).shape)
