# Podman Docker Worker Type Limitations on macOS

## Summary

**There is NO reliable way to run `podman_pool` as Docker worker type on macOS.**

This is due to fundamental architectural differences in how Podman runs on macOS versus Linux.

---

## The macOS Podman Architecture Problem

```
┌─────────────────────────────────────┐
│         macOS Host                   │
│                                      │
│  ┌────────────────────────────┐     │
│  │   Linux VM (QEMU)          │     │
│  │                            │     │
│  │  • Podman daemon           │     │
│  │  • Containers run here     │     │
│  │  • Socket: /run/podman/... │     │
│  │                            │     │
│  └────────────────────────────┘     │
│         ↕ (socket proxy)            │
│  /var/folders/.../podman.sock       │
│  (proxied socket on macOS)          │
└─────────────────────────────────────┘
```

**Key Point**: On macOS, Podman runs inside a Linux virtual machine (QEMU), not natively like on Linux. The socket is proxied to the macOS host, but filesystem operations are limited.

---

## Why Each Approach Fails on macOS

### ❌ Approach 1: Mount socket into container

**Attempted**:
```yaml
volumes:
  - "{{ $PODMAN_SOCKET_PATH }}:/run/podman/podman.sock"
```

**Error**: 
```
500 Server Error: Internal Server Error
making volume mountpoint for volume /var/folders/.../podman.sock: 
operation not supported
```

**Why**: The Podman VM cannot mount arbitrary macOS filesystem paths (especially socket files) into containers. The VM's filesystem is isolated from the macOS host filesystem.

---

### ❌ Approach 2: Use socket without mounting

**Attempted**:
```yaml
env:
  CONTAINER_HOST: "unix:///var/folders/.../podman.sock"
```

**Error**:
```
no such file or directory
```

**Why**: The socket path `/var/folders/...` exists on macOS host, but not inside the worker container. Without mounting, the path doesn't exist.

---

### ❌ Approach 3: Podman CLI in container

**Attempted**:
```dockerfile
RUN apt-get install -y podman
```

**Error**:
```
cannot connect to Podman socket
dial unix /var/folders/.../podman.sock: connect: no such file or directory
```

**Why**: Even with Podman CLI installed in the container, it still needs access to the socket file, which brings us back to Approach 1 and 2 problems.

---

## Linux vs macOS Comparison

| Aspect | Linux | macOS |
|--------|-------|-------|
| **Podman runs** | Natively on host | Inside Linux VM |
| **Socket location** | Direct: `/run/user/1000/podman/podman.sock` | Proxied: `/var/folders/.../podman.sock` |
| **Mount socket into container** | ✅ Works perfectly | ❌ VM blocks it |
| **Container can access host paths** | ✅ Yes, direct access | ❌ No, VM isolation |
| **Docker worker type with Podman** | ✅ Works great | ❌ Impossible |

---

## Your ONLY Options on macOS

### Option 1: Use Process Type ✅ (RECOMMENDED)

**`start_podman_child_worker.sh`:**
```bash
#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Use process type for macOS
prefect work-pool create podman_pool --type "process"
prefect work-pool update podman_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
prefect deploy -n launch_podman --pool podman_pool
PREFECT_WORKER_WEBSERVER_PORT=8082 prefect worker start --pool podman_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Podman worker started"
```

**Use original `prefect.yaml`** (already configured for process type)

**How it works**:
- Worker runs directly on macOS host (not in container)
- Flow calls `podman` CLI which is installed on macOS
- Podman CLI connects to Podman VM via socket
- ✅ Works perfectly

**Pros**:
- ✅ Simple setup
- ✅ Works reliably on macOS
- ✅ No VM complications
- ✅ Same flow code works on Linux too

**Cons**:
- Less isolation (worker runs on host, not in container)
- But for development/testing, this is fine

---

### Option 2: Use Docker Desktop Instead ✅

**`start_docker_child_worker.sh`:**
```bash
#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
export CONTAINER_WORK_DIR=$CONTAINER_WORK_DIR
prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Use Docker instead of Podman
prefect work-pool create docker_pool --type "docker"
prefect work-pool update docker_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
prefect deploy -n launch_docker --pool docker_pool --prefect-file prefect-docker.yaml
PREFECT_WORKER_WEBSERVER_PORT=8081 prefect worker start --pool docker_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Docker worker started"
```

**How it works**:
- Docker Desktop runs containers natively on macOS
- No VM complications like Podman
- ✅ Works perfectly

**When to use**:
- When you need Docker worker type features
- When testing Docker-specific workflows
- When you want better macOS integration

---

### Option 3: Test on Linux ✅

Deploy to a Linux machine (like NSLS-II) where Docker worker type with Podman works:

**Linux `start_podman_child_worker.sh`:**
```bash
#!/bin/bash
source .env

export PREFECT_WORK_DIR=$PREFECT_WORK_DIR
export CONTAINER_WORK_DIR=$CONTAINER_WORK_DIR

# On Linux, Docker worker type works with Podman!
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock

prefect config set PREFECT_API_URL=$PREFECT_API_URL

prefect work-pool create podman_pool --type "docker"
prefect work-pool update podman_pool --concurrency-limit $PREFECT_WORK_POOL_CONCURRENCY
prefect deploy -n launch_podman --pool podman_pool --prefect-file prefect-podman.yaml
PREFECT_WORKER_WEBSERVER_PORT=8082 prefect worker start --pool podman_pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck

echo "Podman worker started"
```

**Linux `prefect-podman.yaml`:**
```yaml
job_variables:
  image: "prefect-with-docker:latest"
  volumes:
    # On Linux, this works!
    - "/run/user/1000/podman/podman.sock:/var/run/docker.sock"
    - "{{ $PREFECT_WORK_DIR }}:{{ $CONTAINER_WORK_DIR }}"
  env:
    DOCKER_HOST: "unix:///var/run/docker.sock"
```

**✅ This works perfectly on Linux!**

---

## Recommended Deployment Strategy

### Development Environment (macOS)
```bash
# Use process type for Podman
prefect work-pool create podman_pool --type "process"

# OR use Docker Desktop
prefect work-pool create docker_pool --type "docker"
```

### Production Environment (Linux - NSLS-II)
```bash
# Enable Podman socket
systemctl --user enable --now podman.socket

# Use Docker worker type with Podman
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
prefect work-pool create podman_pool --type "docker"
```

### Same Flow Code for Both!
Your flow code (`launch_podman`) works the same way on both platforms. Only the worker configuration changes.

---

## Summary Table

| Setup | macOS | Linux | Recommended For |
|-------|-------|-------|-----------------|
| **Process type + Podman** | ✅ Works | ✅ Works | Development, simple deployments |
| **Docker type + Podman + socket** | ❌ Fails | ✅ Works | Production on Linux |
| **Docker type + Docker Desktop** | ✅ Works | ✅ Works | Cross-platform development |

---

## Bottom Line

**On macOS**: 
- Use **process type** for Podman flows
- OR use **Docker Desktop** with docker type
- **DO NOT** try to use docker worker type with Podman

**On Linux**: 
- Use **docker type** with Podman socket
- Works perfectly with proper socket mounting

**The Same Flow Code Works on Both Platforms!** Only the worker configuration needs to change.

---

## Technical Explanation: Why the Limitation Exists

### Docker on macOS
Docker Desktop uses a lightweight Linux VM with optimized filesystem sharing and networking. Docker Inc. has invested heavily in making this transparent to users.

### Podman on macOS
Podman uses a standard QEMU Linux VM. The VM filesystem is more isolated from the macOS host. Socket files and special filesystem operations don't translate well across the VM boundary.

### The Core Issue
When Prefect (using Docker worker type) tries to:
1. Talk to Podman API via socket ✅ (this works via proxy)
2. Mount that socket into a container ❌ (this fails - VM can't mount macOS paths)

The socket needs to be accessible from **inside the container**, but the VM's filesystem isolation prevents this on macOS.

---

## Questions?

**Q: Will this ever work on macOS?**  
A: Unlikely, unless Podman dramatically changes its macOS architecture to match Docker Desktop's approach.

**Q: Does this mean Podman is bad on macOS?**  
A: No! Podman works fine on macOS for normal container operations. The limitation is specific to the "Docker-in-Docker" pattern (running containers from within containers).

**Q: Should I switch from Podman to Docker?**  
A: For macOS development, Docker Desktop might be easier. For Linux production, Podman is excellent and often preferred in enterprise environments.

**Q: Can I develop on macOS and deploy on Linux?**  
A: Yes! Use process type on macOS for development, and docker type on Linux for production. Your flow code stays the same.

---

## File Information

**Created**: November 4, 2025  
**Topic**: Prefect + Podman + macOS Limitations  
**Conclusion**: Use process worker type on macOS, docker worker type on Linux
