#!/bin/bash

# Load environment variables from .env file
source .env

echo "Executing Folder: ${PWD}"

# Start the worker command in the background, capture its PID, and assign the log file
(
    export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
    export PYTHONPATH=$PWD:$PYTHONPATH
    prefect config set PREFECT_API_URL=$PREFECT_API_URL

    # Create log directory if it doesn't exist
    mkdir -p logs

    # Create docker worker pool
    prefect work-pool create docker_pool --type "process" || true
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
    echo "To stop worker, run: kill -9 \$(cat logs/docker_worker_pid.txt)"
)