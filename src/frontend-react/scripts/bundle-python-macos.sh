#!/bin/bash
# Bundle Python with all dependencies for macOS app
# This script should be run from the frontend-react directory

set -e

BUNDLE_DIR="resources/python-bundle"
VENV_DIR="$BUNDLE_DIR/venv"

echo "🐍 Bundling Python for macOS..."

# Create bundle directory
mkdir -p "$BUNDLE_DIR"

# Remove old bundle if it exists
if [ -d "$VENV_DIR" ]; then
  echo "Removing old Python bundle..."
  rm -rf "$VENV_DIR"
fi

# Detect Python installation
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
  PYTHON_CMD="python"
else
  echo "❌ Python not found! Please install Python 3.11 or later."
  exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | awk '{print $2}')
echo "Using Python: $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment at $VENV_DIR..."
$PYTHON_CMD -m venv "$VENV_DIR"

# Activate venv and upgrade pip
echo "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing dependencies..."
if [ -f "python/requirements.txt" ]; then
  "$VENV_DIR/bin/pip" install -r python/requirements.txt
else
  echo "⚠️  requirements.txt not found at python/requirements.txt"
  echo "Installing basic dependencies..."
  "$VENV_DIR/bin/pip" install fastapi uvicorn pydantic
fi

# Verify Python is working
echo "Verifying Python bundle..."
"$VENV_DIR/bin/python" --version

echo "✅ Python bundle created at $BUNDLE_DIR/venv"
echo "Bundle size: $(du -sh "$VENV_DIR" | awk '{print $1}')"
