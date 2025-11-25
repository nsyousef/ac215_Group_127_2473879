#!/bin/bash

# Script to run integration tests for APIManager
# Integration tests make REAL HTTP calls to cloud APIs
# This script:
# 1. Deploys the Modal LLM API
# 2. Sets up API URLs
# 3. Runs integration tests
# 4. Stops the Modal app

set -e

echo "================================================"
echo "Running APIManager Integration Tests"
echo "================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# ==================== Modal LLM Configuration ====================
MODEL_NAME="medgemma-4b"
MAX_TOKENS=70
GPU="H100"
APP_NAME="dermatology-llm-${MODEL_NAME#medgemma-}"
MODAL_USER="tanushkmr2001"

# Track if app was successfully deployed
APP_DEPLOYED=false

# Cleanup function to stop the Modal app
cleanup() {
    if [ "$APP_DEPLOYED" = true ]; then
        echo ""
        echo "================================================"
        echo "Cleaning up: Stopping Modal LLM app..."
        echo "================================================"
        modal app stop "$APP_NAME" 2>/dev/null || echo "⚠️  Could not stop app (it may have already stopped)"
    fi
}

# Set trap to ensure cleanup happens on exit, error, or interruption
trap cleanup EXIT ERR INT TERM

# ==================== Deploy Modal LLM API ====================
echo "Step 1: Deploying Modal LLM API..."
echo "------------------------------------------------"
echo "Model: $MODEL_NAME"
echo "GPU: $GPU"
echo "App Name: $APP_NAME"
echo ""

# Set environment variables for Modal
export MODAL_MODEL_NAME="$MODEL_NAME"
export MODAL_MAX_TOKENS="$MAX_TOKENS"
export MODAL_GPU="$GPU"

# Navigate to LLM directory and deploy
LLM_DIR="../../llm"
if [ -d "$LLM_DIR" ]; then
    cd "$LLM_DIR"

    if modal deploy llm_modal.py; then
        APP_DEPLOYED=true
        echo ""
        echo "✅ Modal LLM API deployed successfully!"
    else
        echo "❌ Modal LLM API deployment failed!"
        exit 1
    fi

    # Return to original directory
    cd "$SCRIPT_DIR"
else
    echo "❌ LLM directory not found at $LLM_DIR"
    exit 1
fi

echo ""

# Wait for endpoints to be ready
echo "Waiting for Modal endpoints to be ready..."
sleep 5

# ==================== API Configuration ====================
echo ""
echo "Step 2: Configuring API URLs..."
echo "------------------------------------------------"

# Cloud ML APIs (already deployed)
export BASE_URL="https://inference-cloud-469023639150.us-east4.run.app"
export TEXT_EMBEDDING_URL="${BASE_URL}/embed-text"
export PREDICTION_URL="${BASE_URL}/predict"

# Modal LLM APIs (just deployed)
export LLM_EXPLAIN_URL="https://${MODAL_USER}--${APP_NAME}-dermatologyllm-explain.modal.run"
export LLM_FOLLOWUP_URL="https://${MODAL_USER}--${APP_NAME}-dermatologyllm-ask-followup.modal.run"

echo "API Configuration:"
echo "  Base URL: $BASE_URL"
echo "  Text Embedding: $TEXT_EMBEDDING_URL"
echo "  Prediction: $PREDICTION_URL"
echo "  LLM Explain: $LLM_EXPLAIN_URL"
echo "  LLM Followup: $LLM_FOLLOWUP_URL"
echo ""

# ==================== Dependency Check ====================

echo "Checking dependencies..."
echo "------------------------------------------------"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed"
    echo "Please install it with: pip install pytest pytest-mock"
    exit 1
fi

# Check if required Python packages are installed
python3 -c "import requests" 2>/dev/null || {
    echo "Error: requests package is not installed"
    echo "Please install it with: pip install requests"
    exit 1
}

python3 -c "import PIL" 2>/dev/null || {
    echo "Error: Pillow package is not installed"
    echo "Please install it with: pip install Pillow"
    exit 1
}

echo "✅ All dependencies OK"
echo ""

# ==================== Run Tests ====================

echo "Step 3: Running integration tests..."
echo "------------------------------------------------"
echo "Note: These tests make real HTTP calls and may take 2-5 minutes"
echo ""

# Run integration tests only
pytest tests/integration.py \
    -v \
    --tb=short \
    --color=yes \
    -m "integration" \
    --durations=10

TEST_EXIT_CODE=$?

# Disable trap temporarily to avoid double cleanup message
trap - EXIT ERR

# ==================== Cleanup ====================

echo ""
echo "Step 4: Cleaning up..."
echo "------------------------------------------------"
cleanup

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "================================================"
    echo "✅ Integration Tests Completed Successfully!"
    echo "================================================"
else
    echo "================================================"
    echo "❌ Integration Tests Failed!"
    echo "================================================"
fi

exit $TEST_EXIT_CODE
