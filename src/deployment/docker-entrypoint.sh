#!/bin/bash
set -e

# Activate virtual environment (created by uv sync)
# This ensures Python can find Pulumi SDK packages (pulumi-gcp, pulumi-docker, etc.)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Execute command
exec "$@"
