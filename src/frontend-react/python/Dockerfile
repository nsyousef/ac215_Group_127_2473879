# Dockerfile for APIManager testing (unit and integration tests)

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-mock==3.12.0 \
    requests==2.32.0 \
    pillow==10.4.0 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    numpy==1.26.4

# Copy source code
COPY api_manager.py pytest.ini /app/
COPY inference_local/ /app/inference_local/

# Copy test files
COPY tests/ /app/tests/

# Copy test runner scripts
COPY run_unit_tests.sh run_integration_tests.sh run_system_tests.sh /app/
RUN chmod +x /app/run_unit_tests.sh /app/run_integration_tests.sh /app/run_system_tests.sh

# Default command
CMD ["./run_unit_tests.sh"]
