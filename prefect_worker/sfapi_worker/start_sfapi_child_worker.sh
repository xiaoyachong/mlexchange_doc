#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Create work pool for job type sfapi (NERSC Superfacility API)
prefect work-pool create sfapi_pool --type "process" || true
prefect work-pool update sfapi_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
prefect deploy -n launch_sfapi --pool sfapi_pool
PREFECT_WORKER_WEBSERVER_PORT=8085 prefect worker start --pool sfapi_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "SFAPI worker started"