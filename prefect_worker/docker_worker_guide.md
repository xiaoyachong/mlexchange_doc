# Prefect Native Docker Worker Migration Guide

This guide shows how to migrate from a process-based Docker worker to Prefect's native Docker work pool type with minimal changes.

## Overview

Currently, all work pools use the "process" type. Prefect supports a native "docker" type that provides better integration and automatic container lifecycle management.

## Changes Required

### 1. Update `prefect.yaml`

Change the docker deployment to use docker-type configuration:

```yaml
- name: launch_docker
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch docker job
  entrypoint: flows/docker/docker_flows.py:launch_docker
  parameters: {}
  work_pool:
    name: docker_pool
    work_queue_name: default-queue
    job_variables:
      image: "{{ image_name }}:{{ image_tag }}"
      auto_remove: true
      stream_output: true
  schedules: []
```

### 2. Update `start_docker_child_worker.sh`

Change the work pool type from "process" to "docker":

```bash
#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Create work pool for job type docker - using native docker type
prefect work-pool create docker_pool --type "docker"
prefect work-pool update docker_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
prefect deploy -n launch_docker --pool docker_pool
PREFECT_WORKER_WEBSERVER_PORT=8081 prefect worker start --pool docker_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Docker worker started"
```

### 3. Update `start_docker_child_worker_background.sh`

Same change - use "docker" type:

```bash
#!/bin/bash

# Load environment variables from .env file
source .env

echo "Executing Folder: ${PWD}"
# Initialize conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Start the worker command in the background, capture its PID, and assign the log file
(
    export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
    export PYTHONPATH=$PWD:$PYTHONPATH
    prefect config set PREFECT_API_URL=$PREFECT_API_URL

    # Create log directory if it doesn't exist
    mkdir -p logs

    # Create docker worker pool - using native docker type
    prefect work-pool create docker_pool --type "docker" || true
    prefect work-pool update docker_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
    prefect deploy -n launch_docker --pool docker_pool
    
    # Start the docker worker with logs that include PID
    PREFECT_WORKER_WEBSERVER_PORT=8081 prefect worker start --pool docker_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck > "process_temp_docker.log" 2>&1 &
    docker_pid=$!
    
    # Rename the log file to include the actual PID of the worker process
    docker_log="logs/docker_worker_${docker_pid}.log"
    mv "process_temp_docker.log" "$docker_log"
    
    # Create a pid file for easy termination later
    echo "$docker_pid" > logs/docker_worker_pid.txt
    
    echo "Started Docker worker with PID: $docker_pid and logging to $docker_log"
    echo "To view logs, use: tail -f $docker_log"
    echo "To stop worker, run: kill \$(cat logs/docker_worker_pid.txt)"
)
```

### 4. Simplify `docker_flows.py` (Optional but Recommended)

With Prefect's native docker support, you can simplify the flow to let Prefect handle the docker execution:

```python
import tempfile

import yaml
from prefect import context, flow
from prefect.states import Failed

from flows.docker.schema import DockerParams
from flows.logger import setup_logger


@flow(name="Docker flow")
async def launch_docker(
    docker_params: DockerParams,
    prev_flow_run_id: str = "",
):
    logger = setup_logger()

    if (
        prev_flow_run_id != ""
        and docker_params.params["io_parameters"]["uid_retrieve"] == ""
    ):
        # Append the previous flow run id to parameters if provided
        docker_params.params["io_parameters"]["uid_retrieve"] = prev_flow_run_id

    current_flow_run_id = str(context.get_run_context().flow_run.id)

    # Append current flow run id
    docker_params.params["io_parameters"]["uid_save"] = current_flow_run_id

    # With native docker work pool, Prefect handles the container execution
    # The parameters are passed via the work pool configuration
    logger.info(f"Launching docker container with image: {docker_params.image_name}:{docker_params.image_tag}")
    logger.info(f"Command: {docker_params.command}")
    logger.info(f"Current flow run ID: {current_flow_run_id}")

    return current_flow_run_id
```

## Key Benefits

1. **Native Docker Support**: Prefect handles Docker container lifecycle automatically
2. **Minimal Code Changes**: Only update work pool type and remove bash script dependency
3. **Better Integration**: Automatic container cleanup and better error handling
4. **Simplified Maintenance**: Less custom bash scripting to maintain
5. **Better Logging**: Prefect provides native logging for Docker containers

## Migration Steps

1. Stop existing Docker workers
2. Delete the old process-type docker_pool: `prefect work-pool delete docker_pool`
3. Apply the changes above
4. Start the new Docker worker using the updated scripts
5. Test with a sample flow run

## Notes

- The bash script `bash_run_docker.sh` can be kept as a fallback but won't be needed with the native docker worker type
- Ensure Docker daemon is accessible to the Prefect worker
- Volume mounts and network configurations are still supported through `job_variables` in `prefect.yaml`

## Rollback

If you need to rollback, simply:
1. Stop the docker worker
2. Delete the docker-type pool
3. Recreate with `--type "process"`
4. Revert code changes
