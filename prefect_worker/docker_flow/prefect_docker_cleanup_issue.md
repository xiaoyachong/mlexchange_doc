# Prefect Docker Container Cleanup Issue - Summary

## The Problem

When cancelling a Prefect flow that launches Docker containers, the **child containers are not cleaned up** and continue running indefinitely.

### Root Cause

1. **SIGTERM kills the process immediately**: When you cancel a flow in Prefect UI, it sends `SIGTERM` to the worker process
2. **`finally` blocks don't execute**: The Python process is terminated with exit code `-15` before the `finally` block can run
3. **Signal handlers don't work**: Python signal handlers (`signal.signal()`) only work in the main thread, but Prefect runs flows in worker threads, causing: `ValueError: signal only works in main thread of the main interpreter`
4. **No resource tracking**: Prefect doesn't track child resources (containers, VMs, etc.) that flows create, so it has no mechanism to clean them up automatically

### Observable Behavior

```
Epoch: 3
*****  memory allocated at epoch 4 is 0
Process for flow run 'married-dodo' exited with status code: -15
```

- Flow is marked as "Cancelled"
- Worker process terminates
- Container (e.g., `761f7afbafa8`) keeps running
- No cleanup logs appear

## Solutions Attempted (All Failed)

### ❌ Solution 1: `finally` Block
```python
finally:
    if container_id:
        await stop_docker_container(container_id)
        await remove_docker_container(container_id)
```
**Why it failed**: `finally` doesn't execute when process receives SIGTERM

### ❌ Solution 2: Signal Handlers (Module Level)
```python
signal.signal(signal.SIGTERM, cleanup_handler)
```
**Why it failed**: `ValueError: signal only works in main thread`

### ❌ Solution 3: Signal Handlers (Flow Level)
```python
@flow
def launch_docker():
    signal.signal(signal.SIGTERM, cleanup_handler)
```
**Why it failed**: Same error - Prefect flows run in worker threads, not main thread

### ❌ Solution 4: Background Monitor Thread
```python
threading.Thread(target=monitor_and_cleanup, daemon=True).start()
```
**Why it failed**: Thread dies with parent process on SIGTERM

### ❌ Solution 5: `asyncio.CancelledError` Handling
```python
except asyncio.CancelledError:
    cleanup_container()
```
**Why it failed**: Exception not raised - process is killed before it can be caught

### ❌ Solution 6: `atexit` Handlers
```python
atexit.register(cleanup_all_containers)
```
**Why it failed**: `atexit` handlers don't run on SIGTERM (only on normal exit)

## Why Prefect Doesn't Have Built-in Solution

**Prefect is a workflow orchestrator, not a resource manager**. It:
- Only tracks flow/task execution, not resources they create
- Assumes you're using infrastructure tools (Kubernetes, Docker Compose) for resource management
- Has no mechanism to track child containers spawned by flows
- Can't reliably clean up when worker processes are killed
