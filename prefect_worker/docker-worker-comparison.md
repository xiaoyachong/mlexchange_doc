# Docker Worker Type Comparison

## What's Actually Happening

With the native Docker worker type, here's the execution flow:

```
┌──────────────────────────────────────────────────────────┐
│ Host Machine                                             │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Prefect Docker Worker Container                    │ │
│  │ (prefecthq/prefect:3.4.2-python3.11)              │ │
│  │                                                    │ │
│  │  - Runs launch_docker flow                        │ │
│  │  - Has /var/run/docker.sock mounted               │ │
│  │  - Executes: docker run autoencoder_image ...     │ │
│  │                                                    │ │
│  └────────────┬───────────────────────────────────────┘ │
│               │ Uses docker socket                      │
│               ▼                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Autoencoder Container (Sibling, not nested!)      │ │
│  │ (your_autoencoder_image:tag)                      │ │
│  │                                                    │ │
│  │  - Runs your ML training code                     │ │
│  │  - Has data volumes mounted                       │ │
│  │                                                    │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Key Point**: The autoencoder container is **NOT** running inside the Prefect container. They're **sibling containers** on the same Docker daemon. This is called "Docker-out-of-Docker" (DooD), not Docker-in-Docker (DinD).

## Complexity Comparison

### Option 1: Native Docker Worker (Current Approach)

**Pros:**
- ✅ Better Prefect integration
- ✅ Automatic container lifecycle management
- ✅ Better logging and monitoring
- ✅ Cleaner error handling
- ✅ No bash script maintenance

**Cons:**
- ❌ Adds one extra container layer (Prefect worker container)
- ❌ More configuration in prefect.yaml
- ❌ Need to manage paths (host vs container)

### Option 2: Process Worker with Bash Script (Original Approach)

**Pros:**
- ✅ Simpler mental model (direct docker commands)
- ✅ No path confusion (always host paths)
- ✅ One less container running

**Cons:**
- ❌ Manual bash script maintenance
- ❌ Less visibility in Prefect UI
- ❌ Manual error handling
- ❌ No automatic cleanup

## Execution Flow Comparison

### Process Worker Flow

```
Parent Flow (Process)
    ↓
launch_docker Flow (Process)
    ↓
bash_run_docker.sh
    ↓
docker run autoencoder_image
```

**Working Directory**: Always uses host path (e.g., `/Users/xiaoyachong/Documents/...`)

### Docker Worker Flow

```
Parent Flow (Process)
    ↓
launch_docker Flow (Docker Container)
    ↓
docker run autoencoder_image (via socket)
```

**Working Directories**:
- Parent Flow: Host path (e.g., `/Users/xiaoyachong/Documents/...`)
- Docker Flow: Container path (e.g., `/mlex_prefect_worker`)

## Path Management

### Process Worker
- All paths are **host paths**
- Single `prefect.yaml` with `{{ $PREFECT_WORK_DIR }}`
- Simpler to understand and debug

### Docker Worker
- Parent flow uses **host paths**
- Docker flow uses **container paths**
- Requires split configuration (`prefect.yaml` + `prefect-docker.yaml`)
- Volume mapping: `/Users/.../mlex_prefect_worker:/mlex_prefect_worker`

## When to Use Each Approach

### Use Process Worker When:
- You have multiple heterogeneous workers (conda, slurm, podman)
- Simplicity and maintainability are priorities
- You don't need advanced container lifecycle features
- Path management should be straightforward
- Team is more familiar with bash/direct Docker commands

### Use Docker Worker When:
- You need Prefect Cloud features (automatic retry, resource limits)
- Running hundreds of concurrent Docker jobs
- Want better integration with Prefect UI
- Need automatic container cleanup and resource management
- Docker is your primary/only execution environment

## Recommendation for MLExchange

**Stick with Process Worker** because:
1. You already have conda, slurm, and podman as process workers
2. Maintaining consistency across all workers is valuable
3. The bash script approach is working well
4. Path management is simpler
5. One less container to manage and debug

**Consider Docker Worker only if**:
- You plan to deprecate other worker types
- You need specific Prefect Cloud features
- You're scaling to hundreds of concurrent jobs

## Migration Considerations

If you do migrate to Docker worker:
- Split `prefect.yaml` into two files for clarity
- Update all path references in documentation
- Test thoroughly with parent flow orchestration
- Ensure Docker socket permissions are correct
- Update monitoring/logging scripts to handle container-based workers

## Successful Migration Steps

### 1. Create Custom Prefect Image with Docker CLI

The default `prefecthq/prefect:3.4.2-python3.11` image doesn't include the Docker CLI, which is required for Docker-out-of-Docker (DooD). Create a custom image:

**Dockerfile:**
```dockerfile
FROM prefecthq/prefect:3.4.2-python3.11

# Install Docker CLI
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    curl \
    gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Verify docker CLI is installed
RUN docker --version
```

**Build the image:**
```bash
docker build -t prefect-with-docker:latest .
```

### 2. Update prefect-docker.yaml

Use the custom image and prevent Docker from trying to pull it:

```yaml
name: mlex_prefect_worker_docker
prefect-version: 3.4.2

build: null
push: null

pull:
- prefect.deployments.steps.set_working_directory:
    directory: "/mlex_prefect_worker"

deployments:
- name: launch_docker
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch docker job (native docker worker)
  entrypoint: flows/docker/docker_flows.py:launch_docker
  parameters: {}
  work_pool:
    name: docker_pool
    work_queue_name: default-queue
    job_variables:
      image: "prefect-with-docker:latest"  # Use custom image
      image_pull_policy: "Never"  # Don't try to pull from registry
      volumes:
        - "/var/run/docker.sock:/var/run/docker.sock"
        - "/Users/xiaoyachong/Documents/3RSE/mlex_prefect_worker:/mlex_prefect_worker"
      env:
        PYTHONPATH: "/mlex_prefect_worker"
      auto_remove: true
      stream_output: true
  schedules: []
```

### 3. Update docker_flows.py

Handle path translations between container and host:

```python
import tempfile
import os
import subprocess
import yaml
from prefect import context, flow
from prefect.states import Failed

from flows.docker.schema import DockerParams
from flows.logger import setup_logger


@flow(name="Docker flow")
def launch_docker(
    docker_params: DockerParams,
    prev_flow_run_id: str = "",
):
    logger = setup_logger()

    if (
        prev_flow_run_id != ""
        and docker_params.params["io_parameters"]["uid_retrieve"] == ""
    ):
        docker_params.params["io_parameters"]["uid_retrieve"] = prev_flow_run_id

    current_flow_run_id = str(context.get_run_context().flow_run.id)
    docker_params.params["io_parameters"]["uid_save"] = current_flow_run_id

    # Create temporary file in the mounted directory so it's accessible from host
    temp_dir = "/mlex_prefect_worker/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create temporary file for parameters
    temp_file_fd, temp_path = tempfile.mkstemp(suffix=".yaml", dir=temp_dir, text=True)
    try:
        with os.fdopen(temp_file_fd, 'w') as f:
            yaml.dump(docker_params.params, f)
        
        logger.info(f"Parameters file: {temp_path}")
        
        # Convert container path to host path for Docker volume mounting
        # /mlex_prefect_worker/tmp/xyz.yaml -> /Users/xiaoyachong/.../tmp/xyz.yaml
        host_temp_path = temp_path.replace(
            "/mlex_prefect_worker", 
            "/Users/xiaoyachong/Documents/3RSE/mlex_prefect_worker"
        )

        # Build docker command directly (no bash script needed)
        docker_cmd = ["docker", "run", "--rm"]
        
        # Add volumes
        volumes = docker_params.volumes + [f"{host_temp_path}:/app/work/config/params.yaml"]
        for volume in volumes:
            docker_cmd.extend(["-v", volume])
        
        # Add network
        if docker_params.network:
            docker_cmd.extend(["--network", docker_params.network])
        
        # Add env vars
        for key, value in docker_params.env_vars.items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        
        # Add image
        docker_cmd.append(f"{docker_params.image_name}:{docker_params.image_tag}")
        
        # Add command with params file path
        command_parts = docker_params.command.split()
        command_parts.append("/app/work/config/params.yaml")
        docker_cmd.extend(command_parts)
        
        logger.info(f"Launching: {' '.join(docker_cmd)}")
        
        # Run docker command directly
        result = subprocess.run(
            docker_cmd,
            check=False,
            capture_output=True,
            text=True
        )
        
        # Log output
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

        if result.returncode != 0:
            return Failed(message="Docker command failed")

        return current_flow_run_id
        
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        logger.error(f"Error launching docker: {str(e)}")
        return Failed(message=f"Docker command failed: {str(e)}")
```

### 4. Keep prefect.yaml for Process Workers

The original `prefect.yaml` remains unchanged and handles all process-based workers (parent, conda, slurm, podman). Only the Docker worker uses `prefect-docker.yaml`:

```yaml
# prefect.yaml - Process workers only
name: mlex_prefect_worker
prefect-version: 3.4.2

build: null
push: null

# Default pull steps use host path
pull:
- prefect.deployments.steps.set_working_directory:
    directory: '{{ $PREFECT_WORK_DIR }}'

deployments:
- name: launch_parent_flow
  # ... parent flow config
  
- name: launch_conda
  # ... conda config
  
- name: launch_slurm
  # ... slurm config
  
- name: launch_podman
  # ... podman config

# Note: launch_docker is REMOVED from this file
# It's now in prefect-docker.yaml
```

### 5. Key Path Translation Issue

When running inside a Docker container, paths need translation:
- **Container sees**: `/mlex_prefect_worker/tmp/file.yaml`
- **Host actually has**: `/Users/xiaoyachong/Documents/3RSE/mlex_prefect_worker/tmp/file.yaml`
- **Child container needs**: Host path for volume mounts

The flow creates temp files in `/mlex_prefect_worker/tmp` (container path), then translates to host path when mounting volumes for child containers.

### 6. Deployment Commands

```bash
# Build custom image
docker build -t prefect-with-docker:latest .

# Deploy process workers (uses prefect.yaml by default)
prefect deploy -n launch_parent_flow --pool parent_pool
prefect deploy -n launch_conda --pool conda_pool
prefect deploy -n launch_slurm --pool slurm_pool
prefect deploy -n launch_podman --pool podman_pool

# Deploy docker worker (uses prefect-docker.yaml explicitly)
prefect work-pool create docker_pool --type "docker"
prefect deploy -n launch_docker --pool docker_pool --prefect-file prefect-docker.yaml

# Start docker worker
PREFECT_WORKER_WEBSERVER_PORT=8081 prefect worker start \
    --pool docker_pool \
    --limit 4 \
    --with-healthcheck
```

### 7. Verification

Check that everything works:

```bash
# Test parent flow calling docker worker
prefect deployment run 'Parent flow/launch_parent_flow' \
  --params '{
    "params_list": [{
      "model_name": "pytorch_autoencoder",
      "task_name": "train",
      "params": {"io_parameters": {}}
    }]
  }'
```

### Common Issues and Solutions

**Issue**: "docker: command not found" in worker container
- **Solution**: Use custom image with Docker CLI installed

**Issue**: "No such file or directory" for temp YAML files
- **Solution**: Create temp files in mounted directory (`/mlex_prefect_worker/tmp`)

**Issue**: Child containers can't access temp files
- **Solution**: Translate container paths to host paths for volume mounts

**Issue**: Image pull errors
- **Solution**: Add `image_pull_policy: "Never"` in prefect-docker.yaml