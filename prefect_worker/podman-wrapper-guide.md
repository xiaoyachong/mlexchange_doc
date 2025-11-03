# Podman Worker with Docker Wrapper - Migration Guide

This guide shows how to add a Podman worker using a Docker-to-Podman wrapper for minimal code changes.

## Overview

Instead of duplicating Docker code, we create a wrapper script that translates `docker` commands to `podman` commands transparently. This allows complete code reuse.

## Files to Create

### 1. Create `Dockerfile.podman`

```dockerfile
FROM prefecthq/prefect:3.4.2-python3.11

# Install Podman CLI
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    curl \
    gnupg && \
    echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libpod:/stable/Debian_12/ /" | tee /etc/apt/sources.list.d/devel:kubic:libpod:stable.list && \
    curl -fsSL https://download.opensuse.org/repositories/devel:kubic:libpod:stable/Debian_12/Release.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/devel_kubic_libpod_stable.gpg && \
    apt-get update && \
    apt-get install -y podman && \
    rm -rf /var/lib/apt/lists/*

# Create docker-to-podman wrapper script
RUN echo '#!/bin/bash' > /usr/local/bin/docker && \
    echo '# Docker-to-Podman wrapper' >> /usr/local/bin/docker && \
    echo '# This script translates docker commands to podman commands transparently' >> /usr/local/bin/docker && \
    echo 'exec podman "$@"' >> /usr/local/bin/docker && \
    chmod +x /usr/local/bin/docker

# Verify both CLIs work
RUN podman --version
RUN docker --version  # Should call podman via wrapper
```

### 2. Create `prefect-podman.yaml`

```yaml
name: mlex_prefect_worker_podman
prefect-version: 3.4.2

build: null
push: null

# Pull steps for Podman worker - uses container path
pull:
- prefect.deployments.steps.set_working_directory:
    directory: "/mlex_prefect_worker"

deployments:
- name: launch_podman
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch podman job (using docker wrapper)
  entrypoint: flows/docker/docker_flows.py:launch_docker  # Reuse docker flow code!
  parameters: {}
  work_pool:
    name: podman_pool
    work_queue_name: default-queue
    job_variables:
      image: "prefect-with-podman:latest"
      image_pull_policy: "Never"
      volumes:
        # Map Podman socket to Docker socket location
        - "/run/podman/podman.sock:/var/run/docker.sock"
        - "/Users/xiaoyachong/Documents/3RSE/mlex_prefect_worker:/mlex_prefect_worker"
      env:
        PYTHONPATH: "/mlex_prefect_worker"
      auto_remove: true
      stream_output: true
  schedules: []
```

**Key insight**: We map the Podman socket (`/run/podman/podman.sock`) to the Docker socket location (`/var/run/docker.sock`) inside the container. This way, the code thinks it's using Docker but actually uses Podman.

### 3. Modify `start_podman_child_worker.sh`

Update your existing file with these minimal changes:

```bash
#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Create work pool for job type podman - CHANGED: use "docker" type for wrapper
prefect work-pool create podman_pool --type "docker"
prefect work-pool update podman_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
# CHANGED: add --prefect-file flag to use prefect-podman.yaml
prefect deploy -n launch_podman --pool podman_pool --prefect-file prefect-podman.yaml
PREFECT_WORKER_WEBSERVER_PORT=8082 prefect worker start --pool podman_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Podman worker started"
```

**Only 2 lines changed:**
1. Line 7: `--type "process"` → `--type "docker"`
2. Line 9: Add `--prefect-file prefect-podman.yaml`

## Files to Modify

### 4. Update `prefect.yaml` (Comment out old podman deployment)

```yaml
name: mlex_prefect_worker
prefect-version: 3.4.2

build: null
push: null

# Default pull steps use host path
pull:
- prefect.deployments.steps.set_working_directory:
    directory: '{{ $PREFECT_WORK_DIR }}'

deployments:
- name: launch_conda
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch conda environment
  entrypoint: flows/conda/conda_flows.py:launch_conda
  parameters: {}
  work_pool:
    name: conda_pool
    work_queue_name: default-queue
    job_variables: {}
  schedules: []

- name: launch_slurm
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch slurm job
  entrypoint: flows/slurm/slurm_flows.py:launch_slurm
  parameters: {}
  work_pool:
    name: slurm_pool
    work_queue_name: default-queue
    job_variables: {}
  schedules: []

# Note: launch_podman moved to prefect-podman.yaml (container-based)
# Note: launch_docker moved to prefect-docker.yaml (container-based)

- name: launch_parent_flow
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch parent flow
  entrypoint: flows/parent_flow.py:launch_parent_flow
  parameters: {}
  work_pool:
    name: parent_pool
    work_queue_name: default-queue
    job_variables: {}
  schedules: []
```

### 5. Update `parent_flow.py` (Minimal changes)

Only need to update the deployment name mapping. Find the section where podman deployment is called and ensure it matches:

```python
elif target_env == FlowType.podman:
    # Prepare podman parameters
    podman_relevant_params = {
        "image_name": algorithm_details["image_name"],
        "image_tag": algorithm_details["image_tag"],
        "command": f"python {python_file}",
        "volumes": job_details["volumes"],
        "network": job_details["network"],
        "env_vars": {},
        "params": params
    }
    # Validate parameters with the schema
    podman_params = PodmanParams(**podman_relevant_params)
    if flow_run_id:
        if "io_parameters" not in podman_params.params:
            podman_params.params["io_parameters"] = {}
        podman_params.params["io_parameters"]["uid_retrieve"] = flow_run_id
    
    # Run the podman deployment with parameters
    deployment_data = {
        "docker_params": podman_params.dict(),  # Note: using docker_params key!
        "prev_flow_run_id": flow_run_id
    }
    flow_run = await run_deployment(
        name="Docker flow/launch_podman",  # Updated deployment name
        parameters=deployment_data
    )
    
    # Check the status of the flow run
    flow_run = await client.read_flow_run(flow_run.id)
    
    if flow_run.state.is_failed():
        prefect_logger.error(f"Step {i+1} failed with state: {flow_run.state.type}")
        return Failed(message=f"Child flow failed with state: {flow_run.state.type}")
        
    flow_run_id = str(flow_run.id)
```

**Important**: The deployment name changes from `"Podman flow/launch_podman"` to `"Docker flow/launch_podman"` because we're reusing the docker flow code. Also use `"docker_params"` instead of `"podman_params"` in the deployment_data.

### 6. Update `config.yml` (Map NSLS-II to Podman)

```yaml
# MLExchange Job Configuration
# Default HPC type to use
hpc_type: "als"

# Configuration for different job types

# Conda environment settings
conda:
  conda_env_name:
    pytorch_autoencoder: "mlex_pytorch_autoencoders"
    pca: "mlex_dimension_reduction_pca"
    umap: "mlex_dimension_reduction_umap"
    clustering: "mlex_clustering"

# Docker/Podman container settings
container:
  volumes:
    - "/Users/xiaoyachong/Documents/3RSE/mlex_data_clinic/data/tiled_storage:/tiled_storage"
  network: "mle_net"

# Slurm job scheduler settings
slurm:
  num_nodes: 1
  partitions: '["p_cpu1", "p_cpu2"]'
  reservations: '["r_cpu1", "r_cpu2"]'
  max_time: "1:00:00"
  submission_ssh_key: "~/.ssh/id_rsa"
```

Update `parent_flow.py` to map NSLS-II to podman:

```python
@task
def determine_best_environment(hpc_type: str) -> FlowType:
    """
    Determine the best execution environment based on hpc_type.
    """
    logger = get_run_logger()
    
    # Map HPC type to flow type
    hpc_type = hpc_type.lower()
    if hpc_type == "nersc":
        logger.info(f"HPC type is NERSC, selecting SLURM")
        return FlowType.slurm
    elif hpc_type == "nsls-ii":
        logger.info(f"HPC type is NSLS-II, selecting PODMAN (with docker wrapper)")
        return FlowType.podman
    elif hpc_type == "als":
        logger.info(f"HPC type is ALS cluster-ball, selecting DOCKER")
        return FlowType.docker
    elif hpc_type in [ft.value for ft in FlowType]:
        return FlowType(hpc_type)
    else:
        logger.info(f"Unknown HPC type: {hpc_type}, defaulting to CONDA environment")
        return FlowType.conda
```

## Files That Don't Need Changes

- ✅ `flows/docker/docker_flows.py` - **Reused as-is for both Docker and Podman!**
- ✅ `flows/docker/schema.py` - **Reused for both!**
- ✅ `flows/podman/bash_run_podman.sh` - Not used anymore (can keep for reference)
- ✅ `flows/podman/podman_flows.py` - Old process-based flow (can keep for fallback)

## Complete File Structure

```
mlex_prefect_worker/
├── Dockerfile                          # Existing: Docker worker image
├── Dockerfile.podman                   # NEW: Podman worker image with wrapper
├── prefect.yaml                        # MODIFIED: Only conda, slurm, parent
├── prefect-docker.yaml                 # Existing: Docker worker config
├── prefect-podman.yaml                 # NEW: Podman worker config (reuses docker code)
├── start_docker_child_worker_standalone.sh  # Existing
├── start_podman_child_worker.sh        # MODIFIED: 2 lines changed
└── flows/
    ├── docker/
    │   ├── docker_flows.py             # REUSED by both docker and podman!
    │   └── schema.py                   # REUSED by both!
    └── podman/
        ├── podman_flows.py             # OLD: Process-based (keep for fallback)
        └── bash_run_podman.sh          # OLD: Not used (keep for reference)
```

## Deployment Steps

### Step 1: Build the Podman Image

```bash
# Build image with docker wrapper
docker build -f Dockerfile.podman -t prefect-with-podman:latest .

# Verify wrapper works
docker run --rm prefect-with-podman:latest docker --version
# Output should show: podman version X.Y.Z
```

### Step 2: Update Configurations

```bash
# Ensure prefect-podman.yaml is created
# Ensure prefect.yaml is updated (remove old podman deployment)
```

### Step 3: Start the Podman Worker

```bash
# Make script executable
chmod +x start_podman_worker.sh

# Start the worker
./start_podman_worker.sh
```

### Step 4: Test the Setup

```bash
# Test through parent flow
prefect deployment run 'Parent flow/launch_parent_flow' \
  --params '{
    "params_list": [{
      "model_name": "pytorch_autoencoder",
      "task_name": "train",
      "params": {
        "io_parameters": {
          "uid_retrieve": "",
          "uid_save": ""
        }
      }
    }]
  }'
```

## Summary of Changes

| File | Status | Description |
|------|--------|-------------|
| `Dockerfile.podman` | **NEW** | Podman image with docker wrapper script |
| `prefect-podman.yaml` | **NEW** | Podman deployment config (reuses docker flow) |
| `start_podman_child_worker.sh` | **MODIFIED** | Change 2 lines: pool type and add yaml file |
| `prefect.yaml` | **MODIFIED** | Remove old podman deployment |
| `parent_flow.py` | **MODIFIED** | Update deployment name from "Podman flow/launch_podman" to "Docker flow/launch_podman", use "docker_params" |
| `flows/docker/docker_flows.py` | **UNCHANGED** | Reused for both docker and podman! |
| `flows/docker/schema.py` | **UNCHANGED** | Reused for both docker and podman! |

## Total Changes Summary

- **2 new files** to create
- **3 existing files** to modify (2 lines in .sh, comments in prefect.yaml, 1 deployment name in parent_flow.py)
- **2 flow files** completely reused (zero duplication!)

## Key Advantages

1. **Minimal changes** - Only 2 new files, 2 lines in shell script
2. **Zero code duplication** - Docker flow code works for both
3. **Easy maintenance** - Single source of truth for container logic
4. **Transparent** - Code doesn't need to know about Podman
5. **Flexible** - Can easily switch between Docker and Podman

## Troubleshooting

### Issue: "docker: command not found" in podman worker

**Solution**: Verify the wrapper was created in the image:
```bash
docker run --rm prefect-with-podman:latest which docker
# Should show: /usr/local/bin/docker
```

### Issue: Podman socket not found

**Solution**: Verify Podman socket location on your system:
```bash
# Check where podman socket is
ls -la /run/podman/podman.sock
# Or
ls -la /var/run/podman/podman.sock

# Update prefect-podman.yaml with correct path
```

### Issue: Parent flow can't find podman deployment

**Solution**: Check deployment name matches:
```bash
# List all deployments
prefect deployment ls

# Should see "Docker flow/launch_podman"
```

## Rollback Plan

If you need to revert to process-based podman:

1. Stop the podman worker: `kill $(cat logs/podman_worker_pid.txt)`
2. Delete the podman pool: `prefect work-pool delete podman_pool`
3. Restore old deployment in `prefect.yaml`
4. Start old process-based worker: `./start_podman_child_worker.sh`
