# forge_rsoxs_analysis — Python 3.12 venv setup (macOS)

Conda's `(base)` Python is 3.11, and the project needs 3.12. Get 3.12 from Homebrew instead:

```bash
conda deactivate
brew install python@3.12
python3.12 --version
```

That should print `Python 3.12.x`. Then build the venv with it:

```bash
cd ~/Documents/3agentic_scattering/forge_rsoxs_analysis
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

That should now print 3.12.x. Then continue with the install:

```bash
pip install -U pip
pip install --only-binary=:all: numba llvmlite
pip install -e .
```

If `brew` isn't found, install Homebrew first from https://brew.sh. If `python3.12` isn't found after installing, use the full path instead: `/usr/local/bin/python3.12 -m venv .venv`.

## Run the app

With the venv installed, run these from the rsoxs folder:

```bash
cd ~/Documents/3agentic_scattering/forge_rsoxs_analysis

# 1. Install the UI dependencies (one-time; needs Node 18+, check with: node -v)
cd frontend && npm install && cd ..

# 2. Stop the old app, since it uses the same ports
(cd ../forge_Agentic_AI_Scattering && docker compose down)

# 3. Start Ollama in another terminal, if it isn't already running
ollama serve

# 4. Preflight check: starts nothing, lists anything missing
./start_all.sh --check

# 5. Start the app (Ctrl+C stops it)
./start_all.sh
```

Then open **http://127.0.0.1:5173**.

Next time, you only need:

```bash
cd ~/Documents/3agentic_scattering/forge_rsoxs_analysis
./start_all.sh
```

- **Other tabs:** for the Companion, Dashboard, Graph and vision panels, start with `PRIMARY_API=agentic ./start_all.sh` instead.
- **Browse will be empty** until you ingest data.
- **If anything fails:** logs are in `.run/logs/<service>.log`.
