#!/bin/bash
set -e

# Activate virtual environment
source /app/.venv/bin/activate

# Check if we're in development mode or running tests
if [ "$1" = "pytest" ]; then
    # Run pytest with all arguments passed after "pytest"
    shift
    exec pytest "$@"
elif [ "$1" = "bash" ]; then
    # Interactive shell for debugging
    exec /bin/bash
elif [ "$DEV" = "1" ]; then
    # Development mode - run server
    exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --reload
else
    # Production mode - run server
    exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
fi
