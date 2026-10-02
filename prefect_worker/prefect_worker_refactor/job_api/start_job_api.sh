#!/bin/bash
# Start the Job API (replaces start_parent_worker.sh). Run from the repo root:
#   ./job_api/start_job_api.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f job_api/.env ] && source job_api/.env; set +a
export PYTHONPATH="$PWD:$PWD/worker:${PYTHONPATH:-}"
exec uvicorn job_api.main:app --host "${JOB_API_HOST:-0.0.0.0}" --port "${JOB_API_PORT:-8090}" --workers "${JOB_API_WORKERS:-2}"
