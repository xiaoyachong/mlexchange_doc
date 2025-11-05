#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
export CONTAINER_WORK_DIR=$CONTAINER_WORK_DIR
export PODMAN_SOCKET_PATH=$PODMAN_SOCKET_PATH

# Set Podman socket
export DOCKER_HOST=unix://${PODMAN_SOCKET_PATH}

echo "Using Podman socket at: $DOCKER_HOST"

prefect config set PREFECT_API_URL=$PREFECT_API_URL

prefect work-pool create podman_pool --type "docker"
prefect work-pool update podman_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY

prefect deploy -n launch_podman --pool podman_pool --prefect-file prefect-podman.yaml

PREFECT_WORKER_WEBSERVER_PORT=8082 prefect worker start --pool podman_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Podman worker started"