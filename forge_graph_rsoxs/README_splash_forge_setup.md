# Set up SPLASH[Scattering] (`splash-forge`) on a Mac, using Shelley's model

`splash-share` is a Google Drive folder shared by Alex Hexemer: a full copy of a working SPLASH[Scattering] install — the code (a newer version of `forge_rsoxs_analysis`), the Tiled catalog, the knowledge graph, past agent conversations and reduction results. This guide gets access to it, downloads it with Google Drive for desktop, copies it to a local folder, unpacks it into `~/Documents/3agentic_scattering/splash-forge`, builds the environment, and points it at the `glm-5.3-flash` model on Shelley.

Tested 20 Oct 2026 on an Apple-silicon Mac.

> **Keep it private.** The data includes publisher PDFs. Don't re-share the Drive folder.

---

## 0. Before you start

- **Mac:** Apple silicon, 32 GB RAM or more.
- **Disk:** about **150 GB free**: ~44 GB Drive download + ~44 GB local copy + ~70 GB unpacked. (You can free the download and the copy afterwards; step 7.)
- **Tools** (Homebrew, https://brew.sh):
  ```bash
  brew install python@3.12 node zstd git
  ```
- Leave conda's `(base)` and any other venv: `conda deactivate`, `deactivate`. The app uses its own Python 3.12 `.venv`.

---

## 1. Get access to the shared folder

1. **Get the link from Alex Hexemer** — a Google Drive link to the `splash-share` folder.
2. Open it in a browser **signed in with your LBL Google account** (`xchong@lbl.gov`). If it says you need access, click **Request access** and wait for Alex to approve.
3. **Add it to My Drive** so the desktop app can see it: in the folder view, click the folder name at the top (or right-click `splash-share` under **Shared with me**) → **Organize → Add shortcut** → choose **My Drive** → **Add**.
   This adds only a *shortcut* (the icon gets a small arrow badge). The folder still belongs to Alex.

---

## 2. Install Google Drive for desktop

1. Download it from **https://www.google.com/drive/download/** and install.
2. Open it from the menu bar and **sign in with `xchong@lbl.gov`**.
3. Keep the default **Stream files** mode (Preferences → Google Drive). Files then appear in Finder but only download when needed — or when you make them available offline (next step).
4. In Finder, the folder now shows up under **Locations → Google Drive → My Drive → splash-share**.

Its real path in Terminal (Finder shortcuts resolve here):

```text
/Users/xiaoyachong/Library/CloudStorage/GoogleDrive-xchong@lbl.gov/.shortcut-targets-by-id/16jhhDGeM_XlzLLsSEAjlk0ERuqAfx2rO/splash-share
```

Set it once per Terminal window:

```bash
DRIVE="/Users/xiaoyachong/Library/CloudStorage/GoogleDrive-xchong@lbl.gov/.shortcut-targets-by-id/16jhhDGeM_XlzLLsSEAjlk0ERuqAfx2rO/splash-share"
ls "$DRIVE/2026-10-16"        # HOW_TO_RUN.md  data  linux  mac  CLAUDE.md
```

> **This folder syncs both ways with Google Drive.** Only *read* from it. Don't edit, move, rename or delete anything inside it — with edit rights that changes Alex's shared folder for everyone. A Finder **alias** to it behaves the same way, and Terminal can't `cd` through an alias ("Not a directory"); always use the real path above.

---

## 3. Make it available offline (download)

1. In Finder: **Google Drive → My Drive**, right-click **`splash-share`** → **Make available offline** (in some versions: **Available offline**).
2. Watch the Google Drive menu-bar window: each `splash-data.tar.zst.part-*` should end up **Downloaded** (11 data parts, ~44 GB, plus the small `mac/` and `linux/` files). This can take a while.
3. Check every file arrived intact (read-only):
   ```bash
   cd "$DRIVE/2026-10-16/data" && shasum -a 256 -c SHA256SUMS
   cd "$DRIVE/2026-10-16/mac"  && shasum -a 256 -c SHA256SUMS
   ```
   Every line must say `OK`. A `FAILED` line means that part hasn't finished downloading — wait and check again. Don't delete anything in the Drive folder.

---

## 4. Copy it to the destination folder

Make a local copy that is independent of Google Drive. Use `cp` — **not** a plain Finder drag, which can *move* the folder out of Drive:

```bash
cp -R "$DRIVE" ~/Documents/3agentic_scattering/splash-forge
SRC=~/Documents/3agentic_scattering/splash-forge
ls "$SRC/2026-10-16"
```

From here on everything reads from `$SRC`, so nothing touches Google Drive. (Set `SRC` again in each new Terminal window.)

The downloaded parts now sit in `splash-forge/2026-10-16/`; the next step unpacks the app into the same `splash-forge` folder, next to them. `splash-forge` must not exist before this `cp` (otherwise `cp` creates `splash-forge/splash-share` instead).

Check the copy too:

```bash
cd "$SRC/2026-10-16/data" && shasum -a 256 -c SHA256SUMS
```

You can pause Google Drive for desktop now if you like (menu bar → gear → **Pause syncing**). Pausing only delays syncing; changes inside the Drive folder still upload when it resumes — so keep working only in `$SRC`.

---

## 5. Unpack the code

```bash
cd "$SRC/2026-10-16"
cat mac/splash-code.tar.zst.part-* | zstd -d | tar -xf - -C ~/Documents/3agentic_scattering/
```

Joins, decompresses and unpacks the code into `~/Documents/3agentic_scattering/splash-forge` (the archive's top folder is `splash-forge`, so it lands in the same folder as the copied `2026-10-16/`).

Unpack straight into the final folder. Moving `splash-forge` afterwards needs extra steps (see "Moving it later").

---

## 6. Unpack the data and create the environment

```bash
cd ~/Documents/3agentic_scattering/splash-forge
scripts/share/setup_mac.sh --data "$SRC/2026-10-16/data" \
    --llm-url http://127.0.0.1:8000/v1 --llm-model glm-5.3-flash
```

What it does (it only *reads* `--data`; everything is written inside `splash-forge`, plus `~/.cache/forge`):

1. checks macOS / arm64, Python 3.12, Node, zstd
2. verifies the data parts, then unpacks them (~70 GB): `catalog.db`, `giwaxs.db`, `data/`, `reduction_api/data/`, agent history, …
3. rewrites stored file paths for this folder (`BUNDLE_INFO.json` → `remapped_to`)
4. builds `.venv` (services) and `venv/train` (MLX fine-tuning)
5. installs the UI (`npm ci`)
6. writes `.env` with **new keys of your own** (`TILED_API_KEY`, `ADMIN_API_KEY`, `JUPYTER_TOKEN`) and the model settings
7. makes a local git repo of the shipped code

It ends with `Setup complete.` CBORG is disabled in this copy (`FORGE_DISABLE_CBORG=1`).

Keep the app's local git repo from seeing the downloaded parts:

```bash
echo "2026-10-16/" >> .git/info/exclude
```

`--llm-url http://127.0.0.1:8000/v1 --llm-model glm-5.3-flash` = Shelley's vLLM model through the SSH tunnel in step 7.

---

## 7. Connect to the model on Shelley

Shelley (`shelley.als.lbl.gov`) serves **`glm-5.3-flash`** (GLM-5.3-Flash-NVFP4) with vLLM on port 8000. See `README_shelley_vllm.md` for details.

In a **new Terminal window**, leave it running:

```bash
ssh -N -o ServerAliveInterval=60 -L 8000:localhost:8000 xchong@shelley.als.lbl.gov
```

LBNL password + token; LBL network or VPN may be needed. After login the window shows nothing — that's correct (`-N` = forward only).

Check from another window:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Expected output (real run, 20 Oct 2026):

```text
{"object":"list","data":[{"id":"glm-5.3-flash","object":"model","created":1792530600,"owned_by":"vllm","root":"/model","parent":null,"max_model_len":750000,"permission":[{"id":"modelperm-9309df55ae951cee","object":"model_permission","created":1792530600,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

- `"id":"glm-5.3-flash"` — the model name to use (in `.env` and on the Config page).
- `"owned_by":"vllm"` — it's Shelley's vLLM server answering, not a local program.
- `"max_model_len":750000` — the context length the server allows.
- The `created` and `modelperm-…` values change each time the server restarts; they don't matter.
- The `%` zsh may print right after the JSON just means the output had no final newline.

### If it says `bind [127.0.0.1]:8000: Address already in use`

Something else holds port 8000:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
ps -p <PID> -o command=
```

- An old `ssh` tunnel: test it with the `curl` above; if it doesn't answer, `kill <PID>` and reconnect.
- Another app (seen on 20 Oct: the `forge_Segementation_Finch` Python server): stop it (`kill <PID>`), kill any half-open `ssh` on `[::1]:8000`, reconnect.
- To keep the other app running, tunnel to another port instead (`-L 8001:localhost:8000`), set `SPLASH_LLM_BASE_URL=http://127.0.0.1:8001/v1` in `splash-forge/.env`, then from `splash-forge`:
  ```bash
  .venv/bin/python scripts/share/native_env.py .env --root "$PWD"
  ```

---

## 8. Start the app

Tunnel open (step 7), then in a second window:

```bash
cd ~/Documents/3agentic_scattering/splash-forge
./start_all.sh --with-jupyter
```

Open **http://127.0.0.1:5173**. Ctrl+C stops everything. The first start can take a few minutes.

Stop anything else on the same ports first (Ctrl+C the `forge_rsoxs_analysis` app; `docker compose down` in `forge_Agentic_AI_Scattering`).

### First time only: the admin key

```bash
grep ADMIN_API_KEY ~/Documents/3agentic_scattering/splash-forge/.env
```

Paste the value into **Config → Services → KG admin key**, Save. Never paste it into docs or chats.

### Quick checks

- **Browse:** samples listed.
- **Explore · KG:** the knowledge graph loads.
- **Scattering Agent / Agent:** ask something; the footer should show `glm-5.3-flash`. Replies can be slow (reasoning effort `max`).
- **Jupyter:** JupyterLab opens.

### Every day

```bash
ssh -N -o ServerAliveInterval=60 -L 8000:localhost:8000 xchong@shelley.als.lbl.gov   # window 1
cd ~/Documents/3agentic_scattering/splash-forge && ./start_all.sh --with-jupyter     # window 2
```

---

## 9. Afterwards: free the space (optional)

`splash-forge` no longer needs the downloaded parts or the local copy.

- **Delete the downloaded parts** once the app runs — only the `2026-10-16` subfolder, **not** `splash-forge` itself (that's the app):
  ```bash
  rm -rf ~/Documents/3agentic_scattering/splash-forge/2026-10-16
  ```
  ~44 GB. This is your own copy, not Drive.

- **Free the ~44 GB, keep access:** Finder → right-click `splash-share` (in Google Drive) → **Make available online only**. Nothing online is deleted.
- **Stop syncing it entirely:** at drive.google.com → My Drive → right-click the `splash-share` **shortcut** (arrow badge on the icon) → **Move to trash**. That removes only your shortcut; Alex's folder stays (find it again under **Shared with me**). Then delete any Finder alias to it.
- A Google Drive message "Unsynced items recovered by the system" just means Drive couldn't upload some local file (usually Finder's hidden `.DS_Store`). **Choose destination** → a folder outside Drive, then delete `.DS_Store`.

---

## 10. Restore the git history (optional)

### Why there is no history

The shared code is a `git archive` of one commit — files only, no `.git`. `setup_mac.sh` then makes a **new** local repo with a single commit (`SPLASH share snapshot ebf2ba2e…`, author `splash-share`), so `git status` shows only your own changes. No remote, no history.

The original commit is recorded in `SHARE_COMMIT.txt`, `BUNDLE_INFO.json` (`git_sha`) and `BUNDLE_MANIFEST.txt`:

```text
ebf2ba2e2b9dde19dd58d5fa267b299ad9b354ba
```

It is on GitHub in `als-computing/forge_rsoxs_analysis` (private — needs your GitHub access):

```text
commit ebf2ba2e2b9dde19dd58d5fa267b299ad9b354ba
Author: Alexander Hexemer <ahexemer@lbl.gov>
Date:   Fri Oct 16 23:02:25 2026 -0700

    share scan: skip raw binary chunks; paper text from publisher PDFs is reported, not counted
```

(Its last change before the bundle was made at 23:05; it only touches the sharing tool's secret scan.)

### Put the branch on that commit (files are not changed)

```bash
cd ~/Documents/3agentic_scattering/splash-forge
cp -R .git ../splash-forge-git-backup              # safety copy of git's records

git remote add upstream https://github.com/als-computing/forge_rsoxs_analysis.git
git fetch upstream                                  # downloads history only

git diff --stat ebf2ba2e HEAD                       # snapshot vs commit: expect nothing,
                                                    # or 2 mode-only changes (below)
git reset --soft ebf2ba2e                           # branch -> real commit; files untouched
```

No GitHub access from this Mac but you have a `forge_rsoxs_analysis` clone that contains the commit? Fetch from the clone instead of `git fetch upstream`:

```bash
git fetch ~/Documents/3agentic_scattering/forge_rsoxs_analysis "+refs/remotes/origin/main:refs/tmp/rsoxs-main"
git reset --soft ebf2ba2e
git update-ref -d refs/tmp/rsoxs-main               # drop the pointer to newer commits
```

Then fix the two settings files the data bundle restored as executable (content is identical):

```bash
chmod 644 src/forge/api/.compute_tier.json src/forge/api/.processing_profile.json
git add src/forge/api/.compute_tier.json src/forge/api/.processing_profile.json
git status                                          # nothing to commit, working tree clean
```

Optionally rename the branch to match upstream (`upstream/main`). Only the label changes; you stay at `ebf2ba2e`:

```bash
git branch -m master main
git branch                                          # * main
```

Don't run `git branch -u upstream/main` — with tracking set, `git pull` would bring in the newer commits. For your own changes, use a separate branch (`git switch -c my-work`) and keep `main` at the shared snapshot.

**What this changes:** only `splash-forge/.git` (git's own records: downloaded history, the branch pointer, the remote in `.git/config`). Code and data files are not touched — except the optional `chmod 644` on the two settings files above, which changes only their permission bits. The data (`catalog.db`, `giwaxs.db`, `data/`, `reduction_api/data/`, `.env`) isn't tracked by git at all. When fetching from a local `forge_rsoxs_analysis` clone, that clone is only read.

**Don't `git pull` / `git merge upstream/main`** if you want to stay at `ebf2ba2e`. The `upstream` remote is a record of where the code came from; `git fetch upstream` updates `upstream/*` references but never your files. No tracking branch is set, so a bare `git pull` does nothing.

Undo: `rm -rf .git && cp -R ../splash-forge-git-backup .git`.

### Cleanup (only if git left temporary files)

If a fetch was interrupted, or another tool couldn't delete git's temporary files, these can be left behind. Safe to remove; `rm` may ask `override r--r--r--?` because git stores pack files read-only — answer `y` (or use `rm -f`):

```bash
rm -f .git/stale-*-lock
rm -f .git/objects/pack/tmp_* .git/objects/pack/*.keep
git update-ref -d refs/tmp/rsoxs-main 2>/dev/null
git gc --prune=now                                  # repack, drop unreferenced objects
```

None of this is needed for the app to work; it just tidies `.git`.

### Check the current commit

```bash
git log -1                    # full hash, author, date, message
git log --oneline -1          # ebf2ba2 share scan: skip raw binary chunks; ...
git rev-parse HEAD            # just the hash
git log --oneline -5          # the last five commits
git branch                    # * main (or * master if not renamed)
git remote -v                 # upstream  https://github.com/als-computing/forge_rsoxs_analysis.git
```

Expected (20 Oct 2026):

```text
ebf2ba2 share scan: skip raw binary chunks; paper text from publisher PDFs is reported, not counted
d6e3b17 docs: HOW_TO_RUN.md, a short first-start guide for the shared copy (Mac native, Linux containers)
f660e11 Linux share: build the amd64 image on the Linux box (emulated apt fails on the Mac); init_code_repo.sh
```

---

## Moving it later

`setup_mac.sh` writes the folder's absolute path into `.env`, the venvs and the databases, so a plain `mv` is not enough. Easiest: leave it and add a link (`ln -s <old> <new>`). To really move it (app stopped):

```bash
mv <old>/splash-forge <new parent>/
cd <new parent>/splash-forge
rm -rf .venv venv/train
scripts/share/setup_mac.sh --data "$PWD" --llm-url http://127.0.0.1:8000/v1 --llm-model glm-5.3-flash
.venv/bin/python scripts/share/native_env.py .env --root "$PWD"
```

The data is not unpacked again ("already unpacked here"); paths are remapped, venvs rebuilt, `.env` and keys kept. If the new path is longer, a few old agent tool outputs in the conversation history keep stale paths.

**Don't re-run the full setup with `--data <data folder>` on an existing install** unless you want to reset it — it would re-unpack ~70 GB over your current catalog and results.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `zsh: command not found: python3.12` | `brew install python@3.12` (don't change conda's base Python) |
| `cd: not a directory` on `…/splash-share` | that's a Finder alias; use the real `$DRIVE` path |
| `shasum` line says `FAILED` | part not fully downloaded; wait for Drive, check again |
| `splash-share` not in Finder's My Drive | step 1.3 (Add shortcut) and step 2 (signed in with the LBL account?) |
| "Unsynced items recovered by the system" | Drive couldn't upload a local file (usually `.DS_Store`); see step 9 |
| `Address already in use` on 8000 | see step 7 |
| Agent doesn't answer | tunnel window still open? `curl -s http://127.0.0.1:8000/v1/models` |
| "enter the admin key" / 401 | step 8, admin key |
| a page can't reach a backend | wait a minute; logs in `splash-forge/.run/logs/` |
| `git log` shows only "SPLASH share snapshot" | expected after setup; step 10 restores the real history |
