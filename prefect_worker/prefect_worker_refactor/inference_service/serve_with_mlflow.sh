#!/bin/bash
# Option B: serve ONE registered model with MLflow's built-in model server
# instead of inference_service/main.py.
#
#   ./inference_service/serve_with_mlflow.sh "DLSIA MSDNet" 3 5001
#
# Exposes POST /invocations (MLflow scoring protocol) and GET /ping.
# Pros: no custom code, uses the model's logged signature/environment.
# Cons: one model per process, no auth (put it behind a proxy), no model
#       switching, input must follow MLflow's JSON format, e.g.
#       curl -X POST localhost:5001/invocations -H 'Content-Type: application/json' \
#            -d '{"inputs": [[...]]}'
set -euo pipefail
MODEL_NAME="$1"; MODEL_VERSION="${2:-latest}"; PORT="${3:-5001}"
set -a; [ -f "$(dirname "$0")/.env" ] && source "$(dirname "$0")/.env"; set +a
exec mlflow models serve -m "models:/${MODEL_NAME}/${MODEL_VERSION}" \
    --host 127.0.0.1 --port "$PORT" --env-manager local
