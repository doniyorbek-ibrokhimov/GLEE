#!/bin/bash
# Setup script for SAM 3 (SAM 2.1) virtual environment
# Creates sam3_venv/ with Python 3.12+ and all required dependencies
#
# Usage:
#   bash setup_sam3_env.sh
#   source sam3_venv/bin/activate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/sam3_venv"

echo "=========================================="
echo "SAM 3 Environment Setup"
echo "=========================================="

# Find Python 3.12+
PYTHON=""
for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
            PYTHON="$candidate"
            echo "Using Python: $PYTHON ($version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "WARNING: Python 3.12+ not found, falling back to python3"
    PYTHON="python3"
fi

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, run: rm -rf $VENV_DIR && bash $0"
else
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate and install dependencies
source "$VENV_DIR/bin/activate"

echo "Installing PyTorch 2.7+ with CUDA 12.6..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "Installing ultralytics (SAM 3 integration)..."
pip install ultralytics

echo "Installing additional dependencies..."
pip install opencv-python-headless numpy

echo "Installing class discovery dependencies..."
pip install google-genai python-dotenv

echo ""
echo "=========================================="
echo "SAM 3 environment setup complete!"
echo "=========================================="
echo ""
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Verify with:"
echo "  python -c \"from ultralytics import SAM; print('SAM 3 OK')\""
