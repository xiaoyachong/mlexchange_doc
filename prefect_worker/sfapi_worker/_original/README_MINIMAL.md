# SFAPI Worker - Minimal Changes Package

This package contains only the **NEW** and **MODIFIED** files needed to add SFAPI support to your existing mlex_prefect_worker project.

## Files to Add (New)

### 1. SFAPI Flow Module
```
flows/sfapi/__init__.py          # New empty file
flows/sfapi/schema.py            # New - SFAPI parameter validation
flows/sfapi/sfapi_flows.py       # New - Main SFAPI flow implementation
```

### 2. Worker Scripts
```
start_sfapi_child_worker.sh                    # New - Start SFAPI worker (foreground)
start_sfapi_child_worker_background.sh         # New - Start SFAPI worker (background)
```

### 3. Example
```
examples/run_sfapi_flow.py       # New - Usage examples
```

## Files to Modify (Existing)

### 1. Core Flow Files
```
flows/parent_flow.py             # MODIFY - Add SFAPI routing (line ~240)
flows/utils.py                   # MODIFY - Add FlowType.sfapi enum
```

### 2. Configuration Files  
```
config.yml                       # MODIFY - Add sfapi section
prefect.yaml                     # MODIFY - Add sfapi deployment
pyproject.toml                   # MODIFY - Add sfapi-client dependency
.env.example                     # MODIFY - Add NERSC credential paths
```

### 3. Documentation
```
README.md                        # MODIFY - Add SFAPI instructions
docs/SFAPI_INTEGRATION.md        # New - Detailed integration guide
```

## Quick Install

### Option 1: Manual File Copy
1. Copy new files from `new_files/` to your project
2. Replace modified files from `modified_files/` 
3. Install dependencies: `pip install sfapi-client authlib`

### Option 2: Apply Patches
1. Review `patches/` directory
2. Apply patches selectively: `patch < patches/parent_flow.patch`

### Option 3: Merge Modified Sections Only
See `MERGE_GUIDE.md` for line-by-line changes to existing files.

## Minimal Steps to Get Running

1. **Copy new SFAPI module**:
   ```bash
   cp -r new_files/flows/sfapi flows/
   ```

2. **Add SFAPI to parent_flow.py** (see line 240 in modified_files/parent_flow.py):
   ```python
   elif target_env == FlowType.sfapi:
       # ... SFAPI routing code ...
   ```

3. **Update utils.py** - Add one line to FlowType enum:
   ```python
   class FlowType(str, Enum):
       # ... existing ...
       sfapi = "sfapi"  # Add this line
   ```

4. **Update config.yml** - Add sfapi section:
   ```yaml
   sfapi:
     machine: "perlmutter"
     queue: "realtime"
     # ... see modified_files/config.yml
   ```

5. **Update prefect.yaml** - Add deployment:
   ```yaml
   - name: launch_sfapi
     # ... see modified_files/prefect.yaml
   ```

6. **Install dependencies**:
   ```bash
   pip install sfapi-client>=0.4.0 authlib>=1.2.0
   ```

7. **Configure credentials in .env**:
   ```bash
   PATH_NERSC_CLIENT_ID=/path/to/client_id.txt
   PATH_NERSC_PRI_KEY=/path/to/private_key.json
   ```

8. **Deploy and start**:
   ```bash
   chmod +x start_sfapi_child_worker.sh
   ./start_parent_worker.sh  # In terminal 1
   ./start_sfapi_child_worker.sh  # In terminal 2
   ```

## What Gets Added to parent_flow.py

Around line 240, after the `elif target_env == FlowType.slurm:` block, add:

```python
elif target_env == FlowType.sfapi:
    # Prepare SFAPI parameters for NERSC execution
    prefect_logger.info("Preparing SFAPI job for NERSC")
    
    job_name = f"{model_name.replace(' ', '_')}_{task_name}_{folder_name}"[:50]
    
    sfapi_relevant_params = {
        "job_name": job_name,
        "machine": job_details.get("sfapi_machine", "perlmutter"),
        "queue": job_details.get("sfapi_queue", "realtime"),
        "account": job_details.get("sfapi_account", "als"),
        "constraint": job_details.get("sfapi_constraint", "cpu"),
        "num_nodes": job_details.get("num_nodes", 1),
        "ntasks_per_node": 1,
        "cpus_per_task": 64,
        "max_time": job_details.get("max_time", "0:15:00"),
        "exclusive": job_details.get("sfapi_exclusive", True),
        "image_name": algorithm_details["image_name"],
        "image_tag": algorithm_details["image_tag"],
        "command": f"python {python_file}",
        "volumes": job_details.get("volumes", []),
        "working_dir": job_details.get("sfapi_working_dir", ""),
        "output_dir": job_details.get("sfapi_output_dir", ""),
        "error_dir": job_details.get("sfapi_error_dir", ""),
        "params": params
    }
    
    sfapi_params = SFAPIParams(**sfapi_relevant_params)
    
    if flow_run_id:
        if "io_parameters" not in sfapi_params.params:
            sfapi_params.params["io_parameters"] = {}
        sfapi_params.params["io_parameters"]["uid_retrieve"] = flow_run_id
    
    deployment_data = {
        "sfapi_params": sfapi_params.dict(),
        "prev_flow_run_id": flow_run_id
    }
    flow_run = await run_deployment(
        name="SFAPI flow/launch_sfapi",
        parameters=deployment_data,
        poll_interval=60
    )
    
    if flow_run.state.is_failed():
        raise RuntimeError(f"Child flow failed at step {i+1}")
        
    flow_run_id = str(flow_run.id)
    prefect_logger.info(f"SFAPI job completed successfully: {flow_run_id}")
```

## Import Addition for parent_flow.py

Add this import at the top:
```python
from flows.sfapi.schema import SFAPIParams
```

## What Gets Added to utils.py

1. In the `FlowType` enum (around line 25):
```python
class FlowType(str, Enum):
    podman = "podman"
    conda = "conda"
    slurm = "slurm"
    docker = "docker"
    sfapi = "sfapi"  # ADD THIS LINE
```

2. In `determine_best_environment()` function (around line 80):
```python
def determine_best_environment(hpc_type: str):
    logger = get_run_logger()
    hpc_type = hpc_type.lower()
    
    if hpc_type == "nersc":
        logger.info(f"Worker type is NERSC, selecting SFAPI")
        return FlowType.sfapi  # MODIFY THIS (was slurm)
    elif hpc_type == "nersc-slurm":
        # ADD THIS for backward compatibility
        logger.info(f"Worker type is NERSC-SLURM, selecting SLURM")
        return FlowType.slurm
    # ... rest of function
```

3. In `get_algorithm_details_from_mlflow()` job_details dict (around line 170):
```python
job_details = {
    # ... existing fields ...
    
    # ADD THESE SFAPI FIELDS:
    "sfapi_machine": config.get("sfapi", {}).get("machine", "perlmutter"),
    "sfapi_queue": config.get("sfapi", {}).get("queue", "realtime"),
    "sfapi_account": config.get("sfapi", {}).get("account", "als"),
    "sfapi_constraint": config.get("sfapi", {}).get("constraint", "cpu"),
    "sfapi_working_dir": config.get("sfapi", {}).get("working_dir", ""),
    "sfapi_output_dir": config.get("sfapi", {}).get("output_dir", ""),
    "sfapi_error_dir": config.get("sfapi", {}).get("error_dir", ""),
    "sfapi_exclusive": config.get("sfapi", {}).get("exclusive", True),
    
    # ... existing conda_env field
}
```

## What Gets Added to config.yml

```yaml
# ADD THIS SECTION at the end:
# SFAPI (NERSC Superfacility API) settings
sfapi:
  machine: "perlmutter"  # NERSC machine (perlmutter, cori)
  queue: "realtime"      # SLURM queue/QOS (realtime, debug, preempt)
  account: "als"         # NERSC account to charge
  constraint: "cpu"      # Node constraint (cpu, gpu)
  num_nodes: 1
  ntasks_per_node: 1
  cpus_per_task: 64
  max_time: "0:15:00"    # Format: HH:MM:SS
  exclusive: true        # Request exclusive node access
  working_dir: ""        # Working directory on NERSC (empty = default pscratch)
  output_dir: ""         # Output log directory (empty = default pscratch/job_logs)
  error_dir: ""          # Error log directory (empty = default pscratch/job_logs)
  volumes: []            # Additional volume mounts for container
```

## What Gets Added to prefect.yaml

```yaml
# ADD THIS DEPLOYMENT:
- name: launch_sfapi
  version: 0.1.0
  tags: []
  concurrency_limit: null
  description: Launch SFAPI job on NERSC
  entrypoint: flows/sfapi/sfapi_flows.py:launch_sfapi
  parameters: {}
  work_pool:
    name: sfapi_pool
    work_queue_name: default-queue
    job_variables: {}
  schedules: []
```

## What Gets Added to pyproject.toml

In the `dependencies` array:
```toml
dependencies = [
    "prefect==3.4.2",
    "typer==0.15.4",
    "mlflow==2.22.0",
    "requests>=2.31.0",
    "python-dotenv",
    "sfapi-client>=0.4.0",  # ADD THIS
    "authlib>=1.2.0"         # ADD THIS
]
```

Update version:
```toml
version = "0.3.0"  # CHANGE FROM 0.2.1
```

## What Gets Added to .env.example

```bash
# ADD THESE LINES:
# NERSC SFAPI Configuration
# =========================
# Path to file containing SFAPI client ID
PATH_NERSC_CLIENT_ID=/path/to/nersc_client_id.txt
# Path to file containing SFAPI private key (JSON format)
PATH_NERSC_PRI_KEY=/path/to/nersc_private_key.json
```

## Testing

After installing, test with:
```bash
python examples/run_sfapi_flow.py
```

## Summary

**Absolute minimum changes needed**:
1. Add 3 new files in `flows/sfapi/`
2. Modify 2 lines in `flows/utils.py`
3. Add ~60 lines to `flows/parent_flow.py`
4. Add config sections to 3 config files
5. Install 2 new packages

That's it! The rest is documentation and optional.
