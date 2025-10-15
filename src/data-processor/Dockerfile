FROM python:3.11-slim-bookworm

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/apcomp215.json

RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && \
    chown app:app /app

USER app
WORKDIR /app
COPY --chown=app:app ac215_Group_127_2473879/src/data-processor ./
COPY --chown=app:app secrets/ ./secrets/

RUN uv sync

ENTRYPOINT ["/bin/bash"]
CMD ["-c", "\
    source /app/.venv/bin/activate && \
    python processor_ddi.py && \
    python processor_fitz.py && \
    python processor_skincap.py && \
    python processor_isic.py && \
    python harmonize_labels.py \
"]