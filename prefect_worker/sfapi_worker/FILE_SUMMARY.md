# SFAPI Worker Files - Summary

## Package Structure

```
sfapi_minimal/
├── README_MINIMAL.md              # This guide
├── INSTALL.sh                     # Automated installation script
├── env_additions.txt              # Environment variable additions
│
├── new_files/                     # NEW files to copy to your project
│   ├── flows/sfapi/
│   │   ├── __init__.py           # Empty module file
│   │   ├── schema.py              # SFAPIParams validation (72 lines)
│   │   └── sfapi_flows.py         # Main SFAPI flow (280 lines)
│   ├── examples/
│   │   └── run_sfapi_flow.py      # Usage examples (85 lines)
│   ├── start_sfapi_child_worker.sh              # Worker script (12 lines)
│   └── start_sfapi_child_worker_background.sh   # Background worker (42 lines)
│
├── modified_files/                # MODIFIED files - compare & merge
│   ├── flows/
│   │   ├── parent_flow.py         # Added SFAPI routing (~50 lines added)
│   │   └── utils.py               # Added FlowType.sfapi (~10 lines added)
│   ├── config.yml                 # Added sfapi section (~15 lines added)
│   ├── prefect.yaml               # Added sfapi deployment (~10 lines added)
│   └── pyproject.toml             # Added 2 dependencies (~2 lines added)
│
└── docs/
    └── SFAPI_INTEGRATION.md       # Comprehensive integration guide
```

## File Categories

### Category 1: Copy As-Is (NEW files)
Just copy these to your project:
- `new_files/flows/sfapi/` → `flows/sfapi/`
- `new_files/start_sfapi_child_worker*.sh` → `./`
- `new_files/examples/run_sfapi_flow.py` → `examples/`
- `docs/SFAPI_INTEGRATION.md` → `docs/`

### Category 2: Merge Changes (MODIFIED files)
Compare and merge changes:
- `modified_files/flows/parent_flow.py` with your `flows/parent_flow.py`
- `modified_files/flows/utils.py` with your `flows/utils.py`
- `modified_files/config.yml` with your `config.yml`
- `modified_files/prefect.yaml` with your `prefect.yaml`
- `modified_files/pyproject.toml` with your `pyproject.toml`

### Category 3: Environment Configuration
- Add lines from `env_additions.txt` to your `.env`

## Line Count Summary

**New code**: ~489 lines
- sfapi_flows.py: 280 lines
- schema.py: 72 lines
- run_sfapi_flow.py: 85 lines
- worker scripts: 54 lines

**Modified code**: ~87 lines added across 5 files
- parent_flow.py: +50 lines
- utils.py: +10 lines
- config.yml: +15 lines
- prefect.yaml: +10 lines
- pyproject.toml: +2 lines

**Total new/modified**: ~576 lines

## Installation Options

### Option A: Automated (Recommended)
```bash
# Extract package to your project root
cd /path/to/mlex_prefect_worker
unzip sfapi_minimal.zip
cd sfapi_minimal
./INSTALL.sh
# Then manually merge the modified files
```

### Option B: Manual
```bash
# 1. Copy new files
cp -r new_files/flows/sfapi flows/
cp new_files/start_sfapi_child_worker*.sh .
chmod +x start_sfapi_child_worker*.sh

# 2. Compare and merge modified files
diff flows/parent_flow.py modified_files/flows/parent_flow.py
# Manually merge changes...

# 3. Update .env
cat env_additions.txt >> .env

# 4. Install dependencies
pip install sfapi-client>=0.4.0 authlib>=1.2.0
```

### Option C: Side-by-side Comparison
Use a diff tool to see exactly what changed:
```bash
# Visual diff tools
meld flows/parent_flow.py modified_files/flows/parent_flow.py
# or
vimdiff flows/utils.py modified_files/flows/utils.py
# or
code --diff config.yml modified_files/config.yml
```

## Verification Checklist

After installation:
- [ ] `flows/sfapi/` directory exists with 3 files
- [ ] `start_sfapi_child_worker.sh` is executable
- [ ] `flows/parent_flow.py` has `elif target_env == FlowType.sfapi:` block
- [ ] `flows/utils.py` has `FlowType.sfapi` in enum
- [ ] `config.yml` has `sfapi:` section
- [ ] `prefect.yaml` has `launch_sfapi` deployment
- [ ] `pyproject.toml` lists `sfapi-client` and `authlib`
- [ ] `.env` has `PATH_NERSC_CLIENT_ID` and `PATH_NERSC_PRI_KEY`
- [ ] Dependencies installed: `pip list | grep sfapi-client`

## Quick Test

```bash
# Test imports
python -c "from flows.sfapi.sfapi_flows import launch_sfapi; print('OK')"

# Test schema
python -c "from flows.sfapi.schema import SFAPIParams; print('OK')"

# Test credentials (will fail if not configured, that's OK for now)
python -c "from flows.sfapi.sfapi_flows import create_sfapi_client; print('Credentials needed')"
```

## Next Steps

1. **Configure NERSC credentials** (see docs/SFAPI_INTEGRATION.md)
2. **Update config.yml** with your NERSC account settings
3. **Start workers**:
   ```bash
   ./start_parent_worker.sh
   ./start_sfapi_child_worker.sh
   ```
4. **Submit test job**:
   ```bash
   python examples/run_sfapi_flow.py
   ```

## Support

See `docs/SFAPI_INTEGRATION.md` for:
- Detailed setup instructions
- Troubleshooting guide
- Architecture diagrams
- Advanced usage examples

## What's NOT Included

These files already exist in your project and don't need changes:
- `flows/__init__.py`
- `flows/conda/*` (except __init__.py)
- `flows/docker/*` (except __init__.py)
- `flows/podman/*` (except __init__.py)
- `flows/slurm/*` (except __init__.py)
- `flows/credentials.py`
- `flows/logger.py`
- All other existing project files

## Minimal Integration

If you want the absolute minimum to get SFAPI working:

**Required new files** (3):
1. `flows/sfapi/sfapi_flows.py`
2. `flows/sfapi/schema.py`
3. `flows/sfapi/__init__.py`

**Required changes** (5 files, ~15 key lines):
1. `flows/parent_flow.py` - Add import + routing block
2. `flows/utils.py` - Add enum value + route logic
3. `config.yml` - Add sfapi section
4. `prefect.yaml` - Add deployment
5. `pyproject.toml` - Add 2 dependencies

Everything else (scripts, examples, docs) is optional but helpful.
