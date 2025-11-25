#!/bin/bash

# Run integration tests with Modal API deployment
# This script:
# 1. Deploys the Modal API
# 2. Extracts endpoint URLs
# 3. Runs integration tests
# 4. Stops the Modal app

set -e  # Exit on error
MODEL_NAME="medgemma-4b"
MAX_TOKENS=50
GPU="H100"

APP_NAME="dermatology-llm-${MODEL_NAME#medgemma-}"

# Track if app was successfully deployed
APP_DEPLOYED=false

# Cleanup function to stop the Modal app
cleanup() {
    if [ "$APP_DEPLOYED" = true ]; then
        echo ""
        echo "Cleaning up: Stopping Modal app..."
        modal app stop "$APP_NAME" 2>/dev/null || echo "⚠️  Could not stop app (it may have already stopped)"
    fi
}

# Set trap to ensure cleanup happens on exit, error, or interruption
trap cleanup EXIT ERR INT TERM

echo "================================"
echo "Integration Test Setup"
echo "================================"
echo "Model: $MODEL_NAME"
echo "GPU: $GPU"
echo "App Name: $APP_NAME"
echo ""

# Set environment variables for Modal
export MODAL_MODEL_NAME="$MODEL_NAME"
export MODAL_MAX_TOKENS="$MAX_TOKENS"
export MODAL_GPU="$GPU"

# Step 1: Deploy the Modal app
echo "Step 1: Deploying Modal app..."
echo "--------------------------------"
if modal deploy llm_modal.py; then
    APP_DEPLOYED=true
else
    echo "❌ Deployment failed!"
    # App was not deployed, so no cleanup needed
    exit 1
fi
echo ""
echo "✅ Deployment successful!"
echo ""

# Step 2: Get endpoint URLs
echo "Step 2: Getting endpoint URLs..."
echo "--------------------------------"

# Wait a moment for endpoints to be ready
sleep 3

# Get Modal username/org
MODAL_USER="tanushkmr2001"
EXPLAIN_URL="https://${MODAL_USER}--${APP_NAME}-dermatologyllm-explain.modal.run"
FOLLOWUP_URL="https://${MODAL_USER}--${APP_NAME}-dermatologyllm-ask-followup.modal.run"

echo "Using endpoint URLs:"
echo "  Explain: $EXPLAIN_URL"
echo "  Follow-up: $FOLLOWUP_URL"

export MODAL_API_EXPLAIN_URL="$EXPLAIN_URL"
export MODAL_API_ASK_FOLLOWUP_URL="$FOLLOWUP_URL"
    
# Step 3: Wait for endpoints to be ready
echo ""
echo "Step 3: Waiting for endpoints to be ready..."
echo "--------------------------------"
sleep 5

# Step 4: Run integration tests
echo ""
echo "Step 4: Running integration tests..."
echo "--------------------------------"
pytest tests/test_llm_modal_integration.py -v

TEST_EXIT_CODE=$?

# Disable trap temporarily to avoid double cleanup message
trap - EXIT ERR

# Step 5: Stop the Modal app (cleanup will also run via trap, but we do it explicitly here)
echo ""
echo "Step 5: Stopping Modal app..."
echo "--------------------------------"
cleanup

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "================================"
    echo "✅ Integration tests completed successfully!"
    echo "================================"
else
    echo "================================"
    echo "❌ Integration tests failed!"
    echo "================================"
fi

exit $TEST_EXIT_CODE

