#!/bin/bash

# Run integration tests with mocked dependencies
# No external API required

set -e

echo "================================"
echo "Running Integration Tests"
echo "================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

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
