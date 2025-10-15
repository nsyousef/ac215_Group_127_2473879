#!/bin/bash
set -e  # Exit on any error

# Create log file with timestamp
LOG_FILE="data-processor.log"

{
    echo "Building data-processor Docker container..."
    docker build -t data-processor -f Dockerfile ../../..
    
    echo ""
    echo "Running data-processor..."
    docker run --rm data-processor
    
} 2>&1 | tee "$LOG_FILE"

echo "Done. Log available at: ${LOG_FILE}"
