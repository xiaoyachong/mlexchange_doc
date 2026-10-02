#!/bin/bash
# Install the SFAPI (NERSC Superfacility API) worker into an mlex_prefect_worker checkout.
#
# Usage:  ./INSTALL.sh /path/to/mlex_prefect_worker
#
# Existing files that get replaced are backed up to <target>/backup_sfapi_<timestamp>/.
# The replaced files (parent_flow.py, utils.py, config.yml, prefect.yaml, pyproject.toml)
# are the upstream versions plus the SFAPI additions only. If your copies have diverged,
# review the backup diff printed at the end.

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-}"

if [ -z "$TARGET" ] || [ ! -f "$TARGET/pyproject.toml" ] || [ ! -d "$TARGET/flows" ]; then
    echo "Usage: $0 /path/to/mlex_prefect_worker"
    exit 1
fi
TARGET="$(cd "$TARGET" && pwd)"

backup="$TARGET/backup_sfapi_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/flows"
for f in flows/parent_flow.py flows/utils.py config.yml prefect.yaml pyproject.toml; do
    [ -f "$TARGET/$f" ] && cp "$TARGET/$f" "$backup/$f"
done
echo "Backup written to $backup"

# New module
mkdir -p "$TARGET/flows/sfapi"
cp "$SRC/sfapi_flow/__init__.py" "$SRC/sfapi_flow/schema.py" "$SRC/sfapi_flow/sfapi_flows.py" "$TARGET/flows/sfapi/"

# Modified files
cp "$SRC/parent_flow.py" "$TARGET/flows/parent_flow.py"
cp "$SRC/utils.py"       "$TARGET/flows/utils.py"
cp "$SRC/config.yml" "$SRC/prefect.yaml" "$SRC/pyproject.toml" "$TARGET/"

# Worker scripts
cp "$SRC"/start_sfapi_child_worker*.sh "$TARGET/"
chmod +x "$TARGET"/start_sfapi_child_worker*.sh

# .env additions (append only if missing)
if [ -f "$TARGET/.env" ] && ! grep -q "PATH_NERSC_CLIENT_ID" "$TARGET/.env"; then
    printf "\n" >> "$TARGET/.env"
    cat "$SRC/env_additions.txt" >> "$TARGET/.env"
    echo "Appended NERSC credential placeholders to .env - edit the paths."
fi

echo
echo "Changes vs. backup:"
for f in flows/parent_flow.py flows/utils.py config.yml prefect.yaml pyproject.toml; do
    [ -f "$backup/$f" ] && { diff -q "$backup/$f" "$TARGET/$f" >/dev/null || echo "  modified: $f"; }
done
echo
echo "Next steps:"
echo "  cd $TARGET && pip install ."
echo "  set worker.name: \"nersc\" in config.yml and fill in the sfapi: section"
echo "  ./start_parent_worker.sh            # terminal 1"
echo "  ./start_sfapi_child_worker.sh       # terminal 2"
