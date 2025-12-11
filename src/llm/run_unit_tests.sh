#!/bin/bash

# Run unit tests only (excludes integration and system tests)

set -e

echo "================================"
echo "Running Unit Tests"
echo "================================"
echo ""

pytest tests/unit.py -v

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "Unit tests passed!"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "ERROR: Unit tests failed!"
    echo "================================"
    exit 1
fi
