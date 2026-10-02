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

    # Create SFAPI worker pool
    prefect work-pool create sfapi_pool --type "process" || true
    prefect work-pool update sfapi_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
    prefect deploy -n launch_sfapi --pool sfapi_pool
    
    # Start the SFAPI worker with logs that include PID
    PREFECT_WORKER_WEBSERVER_PORT=8085 prefect worker start --pool sfapi_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck > "process_temp_sfapi.log" 2>&1 &
    sfapi_pid=$!
    
    # Rename the log file to include the actual PID of the worker process
    sfapi_log="logs/sfapi_worker_${sfapi_pid}.log"
    mv "process_temp_sfapi.log" "$sfapi_log"
    
    # Create a pid file for easy termination later
    echo "$sfapi_pid" > logs/sfapi_worker_pid.txt
    
    echo "Started SFAPI worker with PID: $sfapi_pid and logging to $sfapi_log"
    echo "To view logs, use: tail -f $sfapi_log"
    echo "To stop worker, run: kill -9 \$(cat logs/sfapi_worker_pid.txt)"
)