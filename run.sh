#!/usr/bin/env bash
# ==============================================================================
# NovaScientist: One-Click Launch Script
# Starts the Streamlit Web UI on http://localhost:8501
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "🔬 Starting NovaScientist Autonomous Research & Publication Platform"
echo "======================================================================"

# 1. Resolve Python Interpreter (prefer local virtualenv if available)
PYTHON_CMD="python3"
if [ -f "/Users/chundurisushen/.gemini/antigravity/scratch/research-mind-ai/.venv/bin/python3" ]; then
    PYTHON_CMD="/Users/chundurisushen/.gemini/antigravity/scratch/research-mind-ai/.venv/bin/python3"
elif [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python3"
fi

# 2. Verify Python Version >= 3.10
echo "Checking Python environment..."
$PYTHON_CMD -c "
import sys
if sys.version_info < (3, 10):
    sys.exit('Error: Python 3.10 or higher is required. Found: ' + sys.version)
"

# 3. Check for dependencies
echo "Verifying Python dependencies..."
$PYTHON_CMD -m pip install -q -r requirements.txt || true

# 4. Check for Tectonic LaTeX compiler
echo "Checking LaTeX Tectonic engine..."
if command -v tectonic &> /dev/null || [ -f "/opt/homebrew/bin/tectonic" ]; then
    echo "✓ Tectonic binary detected."
else
    echo "⚠ Tectonic binary not found in PATH. Overleaf multi-file packaging and fallback validation will be used."
fi

# 5. Launch Streamlit Web UI
echo "======================================================================"
echo "🚀 Launching NovaScientist Web UI on http://localhost:8501"
echo "======================================================================"

exec $PYTHON_CMD -m streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
