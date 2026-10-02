import os

import pytest

CONFIG = os.path.join(os.path.dirname(__file__), "..", "job_api", "config.yml")


def fake_details(model_name, model_version=None, gpu=False):
    return {
        "model_name": model_name,
        "model_version": model_version or "3",
        "algorithm_version": "0.1.0",
        "image_name": "ghcr.io/mlexchange/mlex_dlsia_segmentation_prototype",
        "image_tag": "0.1.0",
        "source": "",
        "is_gpu_enabled": gpu,
        "python_file_train": "src/train.py",
        "python_file_inference": "src/segment.py",
    }


@pytest.fixture
def config():
    from job_api.routing import load_config

    os.environ.setdefault("TILED_STORAGE_DIR", "/data/tiled")
    os.environ.setdefault("CONTAINER_NETWORK", "mle_net")
    return load_config(CONFIG)
