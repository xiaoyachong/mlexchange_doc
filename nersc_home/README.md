# NERSC Home Recovery and Cache Configuration

This guide documents how to recover accidentally deleted files from NERSC home snapshots and configure Conda, pip, and Hugging Face to use `/pscratch` instead of consuming home-directory quota.

> **User-specific paths:** The commands below use NERSC username `xchong`. Replace the username and paths if adapting this guide for another user.

## 1. Recover the NERSC home directory

If files in the NERSC home directory are accidentally deleted, you may be able to restore them from a recent home snapshot.

### Step 1: Check home directory usage

```bash
du -sh /global/u2/x/xchong
```

For this account, `/global/homes/x/xchong` is a symlink to `/global/u2/x/xchong`. Checking the real path gives the actual directory usage.

### Step 2: Check available snapshots

```bash
cd ~
ls -F .snapshots
```

Example snapshot names (actual availability varies):

```text
2026-09-28/
2026-09-29/
2026-09-30/
2026-10-01/
2026-10-02/
2026-10-03/
2026-10-04/
2026-10-05/
```

Choose a snapshot from **before** the files were deleted. Inspect its contents, including hidden files:

```bash
ls -la ~/.snapshots/2026-10-05/
```

### Step 3: Preview the recovery (dry run)

```bash
rsync -avhn --delete ~/.snapshots/2026-10-05/ ~/
```

| Option | Meaning |
| --- | --- |
| `-a` | Archive mode; preserve file attributes |
| `-v` | Verbose output |
| `-h` | Human-readable sizes |
| `-n` | Dry run; make no changes |
| `--delete` | Remove destination files not present in the snapshot |

**Important:** Review the dry-run output before proceeding. The `--delete` option can remove files created or modified after the snapshot, including work you want to keep. If you only need specific files, restore those files individually instead of synchronizing your entire home directory. Ensure there is enough quota to complete the restore.

### Step 4: Run the recovery

Only after reviewing the dry run and confirming that replacing the current home contents is appropriate:

```bash
rsync -avh --delete ~/.snapshots/2026-10-05/ ~/
```

> **Caution:** This is a potentially destructive whole-home restore. Consider backing up newer files first. Do not blindly run it if the dry run shows unexpected deletions. Snapshot contents and behavior may differ by system.

### Step 5: Verify the recovered home directory

```bash
cd ~
ls -la
du -sh /global/u2/x/xchong
```

Hidden directories may include `.cache`, `.conda`, `.config`, `.local`, and `.jupyter`, depending on what existed in the selected snapshot.

If restoring the home directory was intended to fix a Jupyter problem, fully stop the existing Jupyter server and start a new session after recovery.

---

## 2. Configure Conda, pip, and Hugging Face to use `/pscratch`

The goal is to keep newly created Conda environments, Conda packages, and large pip/Hugging Face caches outside the NERSC home directory to reduce home quota usage.

> **Note:** `/pscratch` is scratch storage and may be subject to purge or retention policies. Do not treat it as a permanent backup location. Re-create disposable caches if necessary and keep important data in an appropriate persistent storage location.

### Step 1: Create directories on `/pscratch`

```bash
mkdir -p /pscratch/sd/x/xchong/conda_envs
mkdir -p /pscratch/sd/x/xchong/conda_pkgs
mkdir -p /pscratch/sd/x/xchong/pip_cache
mkdir -p /pscratch/sd/x/xchong/huggingface_cache/hub
```

### Step 2: Configure Conda environment and package locations

Make sure `conda` is available in your shell before running these commands.

```bash
conda config --add envs_dirs /pscratch/sd/x/xchong/conda_envs
conda config --add pkgs_dirs /pscratch/sd/x/xchong/conda_pkgs
```

Verify:

```bash
conda config --show envs_dirs
conda config --show pkgs_dirs
```

The `/pscratch` paths should appear in the output.

### Step 3: Configure the pip cache

```bash
export PIP_CACHE_DIR=/pscratch/sd/x/xchong/pip_cache
```

Verify:

```bash
echo "$PIP_CACHE_DIR"
pip cache dir
```

Expected location:

```text
/pscratch/sd/x/xchong/pip_cache
```

### Step 4: Configure the Hugging Face cache

```bash
export HF_HOME=/pscratch/sd/x/xchong/huggingface_cache
export HF_HUB_CACHE=/pscratch/sd/x/xchong/huggingface_cache/hub
```

Verify:

```bash
echo "$HF_HOME"
echo "$HF_HUB_CACHE"
```

Expected locations:

```text
/pscratch/sd/x/xchong/huggingface_cache
/pscratch/sd/x/xchong/huggingface_cache/hub
```

### Step 5: Persist pip and Hugging Face settings in Bash

Append the settings to `~/.bashrc`:

```bash
echo 'export PIP_CACHE_DIR=/pscratch/sd/x/xchong/pip_cache' >> ~/.bashrc
echo 'export HF_HOME=/pscratch/sd/x/xchong/huggingface_cache' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/pscratch/sd/x/xchong/huggingface_cache/hub' >> ~/.bashrc
```

Reload your Bash configuration:

```bash
source ~/.bashrc
```

Only append these lines once to avoid duplicate entries. If Jupyter or another service starts without reading `~/.bashrc`, configure the variables in that service's startup environment as well.

### Step 6: Verify all settings

```bash
echo 'Conda environments:'
conda config --show envs_dirs

echo 'Conda packages:'
conda config --show pkgs_dirs

echo 'pip cache:'
echo "$PIP_CACHE_DIR"
pip cache dir

echo 'Hugging Face caches:'
echo "$HF_HOME"
echo "$HF_HUB_CACHE"
```

Expected `/pscratch` locations:

| Purpose | Path |
| --- | --- |
| Conda environments | `/pscratch/sd/x/xchong/conda_envs` |
| Conda packages | `/pscratch/sd/x/xchong/conda_pkgs` |
| pip cache | `/pscratch/sd/x/xchong/pip_cache` |
| Hugging Face home | `/pscratch/sd/x/xchong/huggingface_cache` |
| Hugging Face Hub cache | `/pscratch/sd/x/xchong/huggingface_cache/hub` |

**Existing environments:** Changing Conda configuration does **not** automatically move existing environments. The settings primarily affect future environment creation, package downloads, and caching. Existing cached files are not moved automatically either.
