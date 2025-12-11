#!/bin/bash

# Run integration tests using FastAPI TestClient
# No external API required

set -e

echo "================================"
echo "Running Integration Tests"
echo "================================"
echo ""

pytest tests/integration.py -v -m integration

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "Integration tests passed!"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "ERROR: Integration tests failed!"
    echo "================================"
    exit 1
fi
