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

    # Create podman worker pool
    prefect work-pool create podman_pool --type "process" || true
    prefect work-pool update podman_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
    prefect deploy -n launch_podman --pool podman_pool
    
    # Start the podman worker with logs that include PID
    PREFECT_WORKER_WEBSERVER_PORT=8082 prefect worker start --pool podman_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck > "process_temp_podman.log" 2>&1 &
    podman_pid=$!
    
    # Rename the log file to include the actual PID of the worker process
    podman_log="logs/podman_worker_${podman_pid}.log"
    mv "process_temp_podman.log" "$podman_log"
    
    # Create a pid file for easy termination later
    echo "$podman_pid" > logs/podman_worker_pid.txt
    
    echo "Started Podman worker with PID: $podman_pid and logging to $podman_log"
    echo "To view logs, use: tail -f $podman_log"
    echo "To stop worker, run: kill -9 \$(cat logs/podman_worker_pid.txt)"
)