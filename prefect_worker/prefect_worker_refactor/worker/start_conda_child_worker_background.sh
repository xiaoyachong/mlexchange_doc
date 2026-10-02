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

    # Create conda worker pool
    prefect work-pool create conda_pool --type "process" || true
    prefect work-pool update conda_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
    prefect deploy -n launch_conda --pool conda_pool
    
    # Start the conda worker with logs that include PID
    PREFECT_WORKER_WEBSERVER_PORT=8083 prefect worker start --pool conda_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck > "process_temp_conda.log" 2>&1 &
    conda_pid=$!
    
    # Rename the log file to include the actual PID of the worker process
    conda_log="logs/conda_worker_${conda_pid}.log"
    mv "process_temp_conda.log" "$conda_log"
    
    # Create a pid file for easy termination later
    echo "$conda_pid" > logs/conda_worker_pid.txt
    
    echo "Started Conda worker with PID: $conda_pid and logging to $conda_log"
    echo "To view logs, use: tail -f $conda_log"
    echo "To stop worker, run: kill -9 \$(cat logs/conda_worker_pid.txt)"
)