# Using Prefect Docker Worker Type with Podman

## The Goal

Use `--type "docker"` for `podman_pool` so that Prefect's Docker worker (which uses Docker SDK/API internally) can communicate with Podman.

## Why This Requires Socket Emulation

When you create a pool with `--type "docker"`, Prefect uses:
- Docker Python SDK (docker-py)
- Docker API calls
- Docker socket communication

Prefect's Docker worker doesn't call `docker` CLI - it talks to the Docker daemon via socket/API. So we need Podman to emulate this.

## Solution: Enable Podman Socket

### Step 1: Enable Podman Socket Service

```bash
# For rootless Podman (recommended)
systemctl --user enable --now podman.socket

# Verify it's running
systemctl --user status podman.socket

# Check socket location
ls -l /run/user/$(id -u)/podman/podman.sock
```

### Step 2: Set DOCKER_HOST Environment Variable

Prefect's Docker worker looks for Docker socket at standard locations. Tell it to use Podman's socket:

```bash
# For rootless Podman
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

# Or for system-wide (if running Podman as root)
export DOCKER_HOST=unix:///run/podman/podman.sock
```

### Step 3: Update Your Start Script

Create or modify `start_podman_docker_worker.sh`:

```bash
#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
export CONTAINER_WORK_DIR=$CONTAINER_WORK_DIR

# Point Docker API calls to Podman socket
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Create DOCKER type work pool (will communicate with Podman socket)
prefect work-pool create podman_pool --type "docker"
prefect work-pool update podman_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY

# Deploy using the docker flow
prefect deploy -n launch_podman --pool podman_pool --prefect-file prefect-podman.yaml

PREFECT_WORKER_WEBSERVER_PORT=8082 prefect worker start --pool podman_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Podman worker (using Docker worker type) started"
```

### Step 4: Create prefect-podman.yaml

Similar to your `prefect-docker.yaml` but for Podman:

```yaml
name: mlex_prefect_worker_podman
prefect-version: 3.4.2

build: null
push: null

# Pull steps for Podman worker - uses container path
pull:
- prefect.deployments.steps.set_working_directory:
    directory: "{{ $CONTAINER_WORK_DIR }}"

deployments:
- name: launch_podman
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch podman container (using Docker worker type)
  entrypoint: flows/docker/docker_flows.py:launch_docker
  parameters: {}
  work_pool:
    name: podman_pool
    work_queue_name: default-queue
    job_variables:
      image: "prefect-with-docker:latest"
      image_pull_policy: "Never"
      volumes:
        # Mount Podman socket instead of Docker socket
        - "/run/user/1000/podman/podman.sock:/var/run/docker.sock"
        - "{{ $PREFECT_WORK_DIR }}:{{ $CONTAINER_WORK_DIR }}"
      env:
        PYTHONPATH: "{{ $CONTAINER_WORK_DIR }}"
        CONTAINER_WORK_DIR: "{{ $CONTAINER_WORK_DIR }}"
        PREFECT_WORK_DIR: "{{ $PREFECT_WORK_DIR }}"
        # Tell Docker SDK to use Podman socket
        DOCKER_HOST: "unix:///var/run/docker.sock"
      auto_remove: true
      stream_output: true
  schedules: []
```

### Step 5: Update Your .env File

Add the socket configuration:

```bash
# .env
PREFECT_WORK_POOL_CONCURRENCY=4
PREFECT_WORKER_LIMIT=4
PREFECT_API_URL=http://localhost:4200/api
PREFECT_WORK_DIR=$PWD
CONTAINER_WORK_DIR=/mlex_prefect_worker
CONDA_PATH=/opt/miniconda3

# Podman socket for Docker-type workers
DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

# MLFlow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=<secure password>
```

## Important Notes

### 1. Podman Socket Compatibility

Podman's Docker-compatible API supports most Docker API calls, but not 100%. Most common operations work fine:
- ✅ `docker run`
- ✅ `docker build`
- ✅ `docker pull`
- ✅ `docker ps`
- ✅ Volume mounts
- ✅ Networks
- ⚠️ Some advanced Docker features may not work

### 2. Rootless vs Root

**Rootless Podman (recommended):**
```bash
# Socket location
/run/user/$(id -u)/podman/podman.sock

# Enable socket
systemctl --user enable --now podman.socket
```

**Root Podman:**
```bash
# Socket location
/run/podman/podman.sock

# Enable socket
systemctl enable --now podman.socket
```

### 3. User ID in Socket Path

The socket path includes your user ID. Replace `1000` with your actual UID:

```bash
# Get your user ID
echo $(id -u)

# Update the volume mount in prefect-podman.yaml
volumes:
  - "/run/user/YOUR_UID/podman/podman.sock:/var/run/docker.sock"
```

### 4. SELinux Considerations

If you're on a system with SELinux (like RHEL, Fedora, CentOS):

```bash
# Check SELinux status
getenforce

# If enforcing, you may need to adjust labels
# Option 1: Allow container_t to access the socket
sudo setsebool -P container_manage_cgroup on

# Option 2: Use :Z flag in volume mounts (in prefect-podman.yaml)
volumes:
  - "/run/user/1000/podman/podman.sock:/var/run/docker.sock:Z"
```

## Testing Your Setup

### Test 1: Verify Podman Socket

```bash
# Check if socket is running
systemctl --user status podman.socket

# Test with curl
curl -H "Content-Type: application/json" \
  --unix-socket /run/user/$(id -u)/podman/podman.sock \
  http://localhost/v1.40/info
```

### Test 2: Test with Docker Python SDK

```python
import docker
import os

# Set socket location
os.environ['DOCKER_HOST'] = f'unix:///run/user/{os.getuid()}/podman/podman.sock'

# Try to connect
client = docker.from_env()
print(client.info())
print("Success! Docker SDK can talk to Podman socket")
```

### Test 3: Run a Simple Container

```bash
# Set environment
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

# Use Docker CLI (will talk to Podman)
docker run --rm hello-world

# Or use Podman directly
podman run --rm hello-world
```

### Test 4: Test Prefect Deployment

```bash
# Source environment with DOCKER_HOST
source .env

# Start the worker
./start_podman_docker_worker.sh

# In another terminal, trigger a deployment
prefect deployment run 'Docker flow/launch_podman'
```

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"

**Solution:**
```bash
# Verify socket is running
systemctl --user status podman.socket

# Restart if needed
systemctl --user restart podman.socket

# Check DOCKER_HOST is set
echo $DOCKER_HOST
```

### Issue: "Permission denied accessing socket"

**Solution:**
```bash
# Check socket permissions
ls -l /run/user/$(id -u)/podman/podman.sock

# Ensure your user owns it (should be automatic for rootless)
# If using rootful Podman, add user to podman group
sudo usermod -aG podman $USER
```

### Issue: "API version mismatch"

**Solution:**
```bash
# Podman socket emulates Docker API, but versions may differ
# Set a compatible API version
export DOCKER_API_VERSION=1.40
```

### Issue: SELinux blocks socket access

**Solution:**
```bash
# Temporarily permissive mode for testing
sudo setenforce 0

# For permanent fix, adjust SELinux policies
sudo setsebool -P container_manage_cgroup on
```

## Comparison: Process Worker vs Docker Worker Type

| Aspect | Process Worker (`--type "process"`) | Docker Worker (`--type "docker"`) |
|--------|-------------------------------------|-----------------------------------|
| **Calls Podman via** | CLI (subprocess) | Socket/API |
| **Setup complexity** | Low | Medium (needs socket) |
| **Compatibility** | 100% Podman | ~95% Podman (some API differences) |
| **Performance** | Slightly slower (subprocess overhead) | Faster (direct API) |
| **When to use** | Simple setups, full compatibility | Need Docker API features, better integration |

## Recommendation

If you want to use `--type "docker"` with Podman:

1. **Enable Podman socket** (Step 1)
2. **Set DOCKER_HOST** everywhere (Step 2)
3. **Update your scripts** (Step 3-4)
4. **Test thoroughly** (Testing section)

This will make Prefect's Docker worker talk to Podman's Docker-compatible socket, giving you the benefits of Docker worker type while running Podman underneath.
