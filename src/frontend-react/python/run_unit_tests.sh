#!/bin/bash

# Script to run unit tests for APIManager
# Unit tests use mocked dependencies and do not require network connectivity

set -e

echo "=================================="
echo "Running APIManager Unit Tests"
echo "=================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"


# Run unit tests only (exclude integration tests)
echo "Running unit tests..."
echo ""

pytest tests/unit.py \
    -v \
    --tb=short \
    --color=yes \
    -m "not integration"

echo ""
echo "=================================="
echo "Unit Tests Complete"
echo "=================================="
