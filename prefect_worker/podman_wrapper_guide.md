# Prefect Docker Worker with Podman Wrapper Guide

This guide shows how to use Prefect's native Docker work pool type for Podman jobs by creating a simple wrapper script that translates Docker commands to Podman.

## Overview

Since Podman CLI is compatible with Docker CLI (you can simply replace `docker` with `podman`), we can use Prefect's native "docker" work pool type for Podman by creating a `docker` wrapper script that calls `podman` instead.

## Solution: Create a Docker-to-Podman Wrapper

### 1. Create `docker` wrapper script

Create a file named `docker` (no extension) in a location that will be in the worker's PATH (e.g., `~/bin/docker` or `/usr/local/bin/docker`):

```bash
#!/bin/bash
# Docker-to-Podman wrapper
# This script translates docker commands to podman commands transparently

exec podman "$@"
```

Make it executable:
```bash
chmod +x ~/bin/docker
```

### 2. Update `start_podman_child_worker.sh`

Change to use docker work pool type and add the wrapper to PATH:

```bash
#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Add wrapper directory to PATH so Prefect uses our docker->podman wrapper
export PATH="$HOME/bin:$PATH"

# Create work pool for podman using docker type
prefect work-pool create podman_pool --type "docker"
prefect work-pool update podman_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
prefect deploy -n launch_podman --pool podman_pool
PREFECT_WORKER_WEBSERVER_PORT=8082 prefect worker start --pool podman_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Podman worker started (using docker work pool type)"
```

### 3. Update `start_podman_child_worker_background.sh`

Same changes:

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
    
    # Add wrapper directory to PATH so Prefect uses our docker->podman wrapper
    export PATH="$HOME/bin:$PATH"

    # Create log directory if it doesn't exist
    mkdir -p logs

    # Create podman worker pool using docker type
    prefect work-pool create podman_pool --type "docker" || true
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
    
    echo "Started Podman worker with PID: $podman_pid and logging to $podman_log (using docker work pool type)"
    echo "To view logs, use: tail -f $podman_log"
    echo "To stop worker, run: kill \$(cat logs/podman_worker_pid.txt)"
)
```

### 4. Update `prefect.yaml`

Change the podman deployment to use docker-type configuration:

```yaml
- name: launch_podman
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch podman container (using docker work pool type)
  entrypoint: flows/podman/podman_flows.py:launch_podman
  parameters: {}
  work_pool:
    name: podman_pool
    work_queue_name: default-queue
    job_variables:
      image: "{{ image_name }}:{{ image_tag }}"
      auto_remove: true
      stream_output: true
  schedules: []
```

### 5. Simplify `podman_flows.py` (Optional)

Similar to the docker flow, you can simplify this:

```python
import tempfile

import yaml
from prefect import context, flow
from prefect.states import Failed

from flows.logger import setup_logger
from flows.podman.schema import PodmanParams


@flow(name="Podman flow")
async def launch_podman(
    podman_params: PodmanParams,
    prev_flow_run_id: str = "",
):
    logger = setup_logger()

    if (
        prev_flow_run_id != ""
        and podman_params.params["io_parameters"]["uid_retrieve"] == ""
    ):
        # Append the previous flow run id to parameters if provided
        podman_params.params["io_parameters"]["uid_retrieve"] = prev_flow_run_id

    current_flow_run_id = str(context.get_run_context().flow_run.id)

    # Append current flow run id
    podman_params.params["io_parameters"]["uid_save"] = current_flow_run_id

    # With docker work pool type + podman wrapper, Prefect handles the container execution
    # The docker commands are transparently translated to podman commands
    logger.info(f"Launching podman container with image: {podman_params.image_name}:{podman_params.image_tag}")
    logger.info(f"Command: {podman_params.command}")
    logger.info(f"Current flow run ID: {current_flow_run_id}")

    return current_flow_run_id
```

## Alternative: System-Wide Wrapper

If you prefer a system-wide solution or don't want to modify PATH, you can create an alias or symlink:

### Option A: Create symlink (requires sudo)
```bash
sudo ln -s $(which podman) /usr/local/bin/docker
```

### Option B: Use environment variable
In your worker startup scripts, add:
```bash
export DOCKER_HOST=unix:///run/podman/podman.sock
```
(Note: This requires podman socket to be enabled)

### Option C: More sophisticated wrapper

Create `~/bin/docker` with better error handling:

```bash
#!/bin/bash
# Docker-to-Podman wrapper with error handling

# Check if podman is available
if ! command -v podman &> /dev/null; then
    echo "Error: podman is not installed or not in PATH" >&2
    exit 1
fi

# Log the translation for debugging (optional)
# echo "[Docker→Podman] Running: podman $@" >&2

# Execute podman with all arguments
exec podman "$@"
```

## Setup Instructions

1. **Create the wrapper script:**
   ```bash
   mkdir -p ~/bin
   cat > ~/bin/docker << 'EOF'
   #!/bin/bash
   exec podman "$@"
   EOF
   chmod +x ~/bin/docker
   ```

2. **Verify the wrapper works:**
   ```bash
   export PATH="$HOME/bin:$PATH"
   docker --version  # Should show podman version
   ```

3. **Stop existing Podman workers:**
   ```bash
   kill $(cat logs/podman_worker_pid.txt)
   ```

4. **Delete old work pool:**
   ```bash
   prefect work-pool delete podman_pool
   ```

5. **Apply the changes above**

6. **Start the new Podman worker:**
   ```bash
   ./start_podman_child_worker_background.sh
   ```

7. **Test with a sample flow run**

## Key Benefits

1. **No Prefect Modifications**: Uses native docker work pool type
2. **Minimal Changes**: Only a simple wrapper script needed
3. **Transparent Translation**: All docker commands automatically become podman commands
4. **Full Feature Support**: All docker work pool features work with podman
5. **Easy Maintenance**: Single wrapper script to manage

## Compatibility Notes

- Podman CLI is designed to be compatible with Docker CLI
- Most docker commands work identically with podman
- Some advanced Docker features might have slight differences
- Test thoroughly with your specific use cases

## Troubleshooting

### Wrapper not being used
Check that the wrapper directory is first in PATH:
```bash
which docker  # Should show ~/bin/docker
```

### Permission issues
Podman runs rootless by default, which might have different behavior than Docker:
```bash
# Check podman is working
podman ps
```

### Socket issues
If using socket-based communication:
```bash
systemctl --user enable --now podman.socket
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
```

## Rollback

If you need to rollback:
1. Remove the wrapper: `rm ~/bin/docker`
2. Stop the worker
3. Delete the docker-type pool
4. Recreate with `--type "process"`
5. Revert code changes
