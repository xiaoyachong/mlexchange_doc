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

    # Create slurm worker pool
    prefect work-pool create slurm_pool --type "process" || true
    prefect work-pool update slurm_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
    prefect deploy -n launch_slurm --pool slurm_pool
    
    # Start the slurm worker with logs that include PID
    PREFECT_WORKER_WEBSERVER_PORT=8084 prefect worker start --pool slurm_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck > "process_temp_slurm.log" 2>&1 &
    slurm_pid=$!
    
    # Rename the log file to include the actual PID of the worker process
    slurm_log="logs/slurm_worker_${slurm_pid}.log"
    mv "process_temp_slurm.log" "$slurm_log"
    
    # Create a pid file for easy termination later
    echo "$slurm_pid" > logs/slurm_worker_pid.txt
    
    echo "Started Slurm worker with PID: $slurm_pid and logging to $slurm_log"
    echo "To view logs, use: tail -f $slurm_log"
    echo "To stop worker, run: kill -9 \$(cat logs/slurm_worker_pid.txt)"
)