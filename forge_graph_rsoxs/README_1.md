# forge_rsoxs_analysis — setup and run (macOS)

## 1. Python 3.12 venv

Conda's `(base)` Python is 3.11, and the project needs 3.12. Get 3.12 from Homebrew instead (don't change conda's base Python):

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
pip install -e ".[optics,rsoxs]"
```

- `--only-binary` for numba/llvmlite: on an Intel (x86_64) Python they otherwise try to compile and fail.
- `[optics,rsoxs]` adds `xraydb`, `periodictable` and `kkcalc`. Without them the RSoXS page shows a red "Material ... is under-specified ... (with xraydb installed)" error. For all simulation pages (Grazing, Crystals, Gratings, Mesh) use `pip install -e ".[sim]"` instead.

If `brew` isn't found, install Homebrew first from https://brew.sh. If `python3.12` isn't found after installing, use the full path instead: `/usr/local/bin/python3.12 -m venv .venv`.

To leave the venv: `deactivate` (`conda deactivate` only exits conda).

## 2. Configure `.env`

```bash
cp .env.example .env
```

Set at least:

```bash
OLLAMA_MODEL=qwen3.5:9b
VISION_MODEL=qwen3-vl:8b
VISION_EMBEDDING_MODEL=qwen3-embedding:0.6b
ML_WEIGHTS_NAME=vae_model_weights.npz    # file goes in reduction_api/model_weights/
```

Leave `TILED_API_KEY` empty and `start_all.sh` generates one (or reuse the key from the old repo's `.env`).

## 3. Run the app

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

- **All tabs:** plain `./start_all.sh` reaches both backends (forge and agentic) through the UI; `PRIMARY_API=agentic` is not needed.
- **After installing new Python packages** (e.g. `[optics,rsoxs]`): Ctrl+C and rerun `./start_all.sh`, then click Refresh on the page.
- **If anything fails:** logs are in `.run/logs/<service>.log`.
- zsh prints a harmless `command not found: #` for comment lines if you paste a whole block.

## 4. Data

The catalog starts empty; nothing carries over from the old repo (its `catalog.db` uses Docker-only `/catalog/...` paths — don't copy it).

**Works with no data:** RSoXS, Grazing, Crystals, Gratings, Mesh, GIMaP Predict/WAXS, Agent (needs Ollama), Ingest · KG / Paper · KG / Graph (from PDF papers), Config, Help; Scattering → Select → upload a single frame (TIFF, EDF, CBF, NPY).

**Needs ingested data:** Browse, Data, Companion, Dashboard, Trends, Selections, Kinetics, Embedding Explorer, Scattering's catalog-based tabs.

Ingest from a second terminal while the app runs (the UI cannot bulk-ingest into the catalog; Ingest · KG is for papers):

```bash
# EDF frames
.venv/bin/python scripts/ingest_edf_to_tiled.py --source /path/to/data --dry-run
.venv/bin/python scripts/ingest_edf_to_tiled.py --source /path/to/data

# Simulated RSoXS energy scans (clearly marked synthesized: true)
.venv/bin/python scripts/synthesize_rsoxs_scan_to_tiled.py --dry-run
.venv/bin/python scripts/synthesize_rsoxs_scan_to_tiled.py
```

The 10k samples (`ten_k_Samples/raw_h5*/date/bar/*.h5`) have **no ingest script** in this repo — the old `ingest_ten_k_h5_to_tiled.py` was dropped and would need porting.
