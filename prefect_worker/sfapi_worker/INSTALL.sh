#!/bin/bash
# Quick install script for SFAPI worker support

set -e

echo "==================================================================="
echo "SFAPI Worker Installation Script"
echo "==================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Please run this script from the root of your mlex_prefect_worker project"
    exit 1
fi

echo "Step 1: Copying new SFAPI module files..."
mkdir -p flows/sfapi
cp new_files/flows/sfapi/* flows/sfapi/
echo "  ✓ SFAPI module files copied"

echo ""
echo "Step 2: Copying worker startup scripts..."
cp new_files/start_sfapi_child_worker*.sh .
chmod +x start_sfapi_child_worker*.sh
echo "  ✓ Worker scripts copied and made executable"

echo ""
echo "Step 3: Copying example files..."
mkdir -p examples
cp new_files/examples/run_sfapi_flow.py examples/
echo "  ✓ Example file copied"

echo ""
echo "Step 4: Copying documentation..."
mkdir -p docs
cp docs/SFAPI_INTEGRATION.md docs/
echo "  ✓ Documentation copied"

echo ""
echo "Step 5: Backing up existing configuration files..."
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
cp flows/parent_flow.py "$backup_dir/" 2>/dev/null || echo "  Note: parent_flow.py not found"
cp flows/utils.py "$backup_dir/" 2>/dev/null || echo "  Note: utils.py not found"
cp config.yml "$backup_dir/" 2>/dev/null || echo "  Note: config.yml not found"
cp prefect.yaml "$backup_dir/" 2>/dev/null || echo "  Note: prefect.yaml not found"
cp pyproject.toml "$backup_dir/" 2>/dev/null || echo "  Note: pyproject.toml not found"
echo "  ✓ Backup created in $backup_dir/"

echo ""
echo "==================================================================="
echo "IMPORTANT: Manual steps required"
echo "==================================================================="
echo ""
echo "The following files need manual updates:"
echo ""
echo "1. flows/parent_flow.py"
echo "   - Add import: from flows.sfapi.schema import SFAPIParams"
echo "   - Add SFAPI routing block (see modified_files/flows/parent_flow.py line 240)"
echo ""
echo "2. flows/utils.py"
echo "   - Add FlowType.sfapi to enum"
echo "   - Update determine_best_environment() for 'nersc'"
echo "   - Add SFAPI fields to job_details dict"
echo ""
echo "3. config.yml"
echo "   - Add sfapi configuration section (see modified_files/config.yml)"
echo ""
echo "4. prefect.yaml"
echo "   - Add launch_sfapi deployment (see modified_files/prefect.yaml)"
echo ""
echo "5. pyproject.toml"
echo "   - Add dependencies: sfapi-client>=0.4.0, authlib>=1.2.0"
echo ""
echo "6. .env"
echo "   - Add NERSC credential paths (see env_additions.txt)"
echo ""
echo "==================================================================="
echo "Quick comparison tool:"
echo "==================================================================="
echo ""
echo "Compare your files with the modified versions:"
echo "  diff flows/parent_flow.py modified_files/flows/parent_flow.py"
echo "  diff flows/utils.py modified_files/flows/utils.py"
echo "  diff config.yml modified_files/config.yml"
echo "  diff prefect.yaml modified_files/prefect.yaml"
echo "  diff pyproject.toml modified_files/pyproject.toml"
echo ""
echo "Or use a merge tool:"
echo "  meld flows/parent_flow.py modified_files/flows/parent_flow.py"
echo ""
echo "==================================================================="
echo "After manual updates, run:"
echo "==================================================================="
echo ""
echo "  pip install -e ."
echo "  ./start_sfapi_child_worker.sh"
echo ""
echo "==================================================================="
