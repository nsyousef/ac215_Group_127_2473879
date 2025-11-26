#!/bin/bash

# Run system tests against deployed APIs
# Assumes cloud APIs and Modal LLM API are already deployed

set -e

echo "================================"
echo "Running System Tests"
echo "================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set defaults (can be overridden via environment variables)
MODAL_USER="${MODAL_USER:-tanushkmr2001}"
MODEL_SUFFIX="${MODEL_SUFFIX:-4b}"
APP_NAME="dermatology-llm-${MODEL_SUFFIX}"

# Cloud ML APIs
export BASE_URL="${BASE_URL:-https://inference-cloud-469023639150.us-east4.run.app}"
export TEXT_EMBEDDING_URL="${TEXT_EMBEDDING_URL:-${BASE_URL}/embed-text}"
export PREDICTION_URL="${PREDICTION_URL:-${BASE_URL}/predict}"

# Modal LLM APIs
export LLM_EXPLAIN_URL="${LLM_EXPLAIN_URL:-https://${MODAL_USER}--${APP_NAME}-dermatologyllm-explain.modal.run}"
export LLM_FOLLOWUP_URL="${LLM_FOLLOWUP_URL:-https://${MODAL_USER}--${APP_NAME}-dermatologyllm-ask-followup.modal.run}"

echo "Testing endpoints:"
echo "  Cloud ML: $BASE_URL"
echo "  LLM Explain: $LLM_EXPLAIN_URL"
echo "  LLM Followup: $LLM_FOLLOWUP_URL"
echo ""

pytest tests/system.py -v -m system --durations=10

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ System tests passed!"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "❌ System tests failed!"
    echo "================================"
    exit 1
fi

