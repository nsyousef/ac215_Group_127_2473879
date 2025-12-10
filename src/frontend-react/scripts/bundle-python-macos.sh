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

# Step 1: Install the right PyTorch version for the host architecture
ARCHITECTURE=$(uname -m)
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"

case "$ARCHITECTURE" in
  x86_64)
    # Intel Macs need a downlevel PyTorch build to avoid segfaults
    TORCH_VERSION="2.2.2"
    TORCHVISION_VERSION="0.17.2"
    ;;
  arm64)
    # Apple Silicon Macs can use the newer, M-series verified build
    TORCH_VERSION="2.9.1"
    TORCHVISION_VERSION="0.24.1"
    ;;
  *)
    echo "⚠️  Unknown architecture ($ARCHITECTURE). Defaulting to torch $TORCH_VERSION."
    ;;
esac

echo "Installing PyTorch $TORCH_VERSION for $ARCHITECTURE..."
"$VENV_DIR/bin/pip" install --no-cache-dir \
  torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

# Step 2: Install remaining dependencies from requirements-build.txt (excludes torch & test deps)
# Falls back to requirements-ci.txt if build version doesn't exist
if [ -f "python/requirements-build.txt" ]; then
  echo "Installing remaining dependencies from requirements-build.txt (prod bundle)..."
  "$VENV_DIR/bin/pip" install --no-cache-dir \
    --requirement python/requirements-build.txt
elif [ -f "python/requirements-ci.txt" ]; then
  echo "Installing remaining dependencies from requirements-ci.txt (CI-safe fallback)..."
  "$VENV_DIR/bin/pip" install --no-cache-dir \
    --requirement python/requirements-ci.txt
else
  echo "⚠️  No requirements files found"
  echo "Installing minimal dependencies..."
  "$VENV_DIR/bin/pip" install --no-cache-dir \
    requests pillow modal numpy
fi

# Optimization: Remove unnecessary files to reduce bundle size
echo "Optimizing bundle size..."

# Remove .pyc files and __pycache__ directories
find "$VENV_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$VENV_DIR" -type f -name "*.pyc" -delete
find "$VENV_DIR" -type f -name "*.pyo" -delete

# Remove test files from packages
echo "  Removing test directories..."
find "$VENV_DIR/lib" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
find "$VENV_DIR/lib" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$VENV_DIR/lib" -type d -name "*_test" -exec rm -rf {} + 2>/dev/null || true

# Remove PyTorch test files and unnecessary subdirectories
echo "  Removing PyTorch test and non-essential files..."
if [ -d "$VENV_DIR/lib/python*/site-packages/torch" ]; then
  # Remove test directories (safe to remove)
  find "$VENV_DIR/lib/python"*/site-packages/torch -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
  find "$VENV_DIR/lib/python"*/site-packages/torch -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true

  # Remove bin directory (executables not needed at runtime)
  rm -rf "$VENV_DIR/lib/python"*/site-packages/torch/bin 2>/dev/null || true

  # Remove only specific large cmake files (keep the directory structure)
  find "$VENV_DIR/lib/python"*/site-packages/torch/share/cmake -name "*.cmake" -size +1M -delete 2>/dev/null || true

  echo "  PyTorch optimization complete (kept runtime dependencies)"
fi

# Remove transformers test files if installed
if [ -d "$VENV_DIR/lib/python*/site-packages/transformers" ]; then
  find "$VENV_DIR/lib/python*/site-packages/transformers" -type f -name "test_*.py" -delete
fi

# Strip debug symbols from compiled libraries (.so files)
# DISABLED: Breaks PyTorch on M-series
# echo "  Stripping debug symbols from compiled libraries..."
# find "$VENV_DIR" -type f \( -name "*.so" -o -name "*.dylib" \) ! -path "*/bin/*" -exec strip -x {} + 2>/dev/null || true

# Remove pip cache
rm -rf "$VENV_DIR/lib/python"*/site-packages/pip* 2>/dev/null || true

# Remove setuptools, wheel, and other build tools not needed at runtime
echo "  Removing build tools..."
rm -rf "$VENV_DIR/lib/python"*/site-packages/pip 2>/dev/null || true
rm -rf "$VENV_DIR/lib/python"*/site-packages/setuptools 2>/dev/null || true
rm -rf "$VENV_DIR/lib/python"*/site-packages/wheel 2>/dev/null || true
rm -rf "$VENV_DIR/lib/python"*/site-packages/easy_install.py 2>/dev/null || true

# Verify Python is working
echo "Verifying Python bundle..."
"$VENV_DIR/bin/python" --version

# Export certifi bundle for use at runtime (fixes SSL in production)
CERTIFI_PEM=$("$VENV_DIR/bin/python" - <<'PY'
import certifi
print(certifi.where())
PY
)

if [ -n "$CERTIFI_PEM" ] && [ -f "$CERTIFI_PEM" ]; then
  echo "Bundling certifi CA store..."
  cp "$CERTIFI_PEM" "$BUNDLE_DIR/cacert.pem"
else
  echo "⚠️  Could not locate certifi certificate bundle; HTTPS requests may fail."
fi

FINAL_SIZE=$(du -sh "$VENV_DIR" | awk '{print $1}')
echo "✅ Python bundle created at $BUNDLE_DIR/venv"
echo "Final bundle size: $FINAL_SIZE (optimized)"
