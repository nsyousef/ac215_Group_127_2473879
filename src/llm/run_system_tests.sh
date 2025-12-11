#!/bin/bash

# Run system tests against deployed Modal API
# Assumes the Modal app is already deployed externally

set -e

echo "================================"
echo "Running System Tests"
echo "================================"
echo ""

# Set default endpoint URLs (can be overridden via environment variables)
MODAL_USER="${MODAL_USER:-tanushkmr2001}"
MODEL_SUFFIX="${MODEL_SUFFIX:-4b}"
APP_NAME="dermatology-llm-${MODEL_SUFFIX}"

EXPLAIN_URL="${MODAL_API_EXPLAIN_URL:-https://${MODAL_USER}--${APP_NAME}-dermatologyllm-explain.modal.run}"
FOLLOWUP_URL="${MODAL_API_ASK_FOLLOWUP_URL:-https://${MODAL_USER}--${APP_NAME}-dermatologyllm-ask-followup.modal.run}"

echo "Testing endpoints:"
echo "  Explain: $EXPLAIN_URL"
echo "  Follow-up: $FOLLOWUP_URL"
echo ""

export MODAL_API_EXPLAIN_URL="$EXPLAIN_URL"
export MODAL_API_ASK_FOLLOWUP_URL="$FOLLOWUP_URL"

# Run system tests
pytest tests/system.py -v -m system

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "System tests passed!"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "ERROR: System tests failed!"
    echo "================================"
    exit 1
fi
