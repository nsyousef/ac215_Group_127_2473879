"""
Modal Training with Volume Cache (RECOMMENDED)

This approach pre-loads data from GCS into a Modal Volume, then uses fast local disk I/O.

Setup:
  1. modal run modal_training_volume.py::sync_data_from_gcs  (one-time)
  2. modal run modal_training_volume.py --config-path ...    (training)
"""

import logging
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Get paths - need to mount src/ directory so ml_workflow package structure works
SRC_DIR = Path(__file__).parent.parent  # Goes up to src/
ML_WORKFLOW_DIR = Path(__file__).parent  # ml_workflow directory

# Define the image with all required dependencies AND your local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "torch>=2.0.0",
            "torchvision",
            "transformers>=4.21.0",
            "datasets",
            "Pillow",
            "pandas",
            "numpy",
            "scikit-learn",
            "tqdm",
            "PyYAML",
            "wandb",
            "matplotlib",
            "seaborn",
            "accelerate",
            "safetensors",
            "google-cloud-storage",  # Python GCS client for syncing
            "pyarrow",
            "gcsfs",  # For GCS filesystem access
        ]
    )
    .apt_install(["git"])
    # Mount src/ directory so ml_workflow package structure works
    # This creates /app/src/ml_workflow/ structure
    .add_local_dir(str(SRC_DIR), "/app/src")
)

app = modal.App("skin-disease-training-volume", image=image)


# ============================================================================
# STEP 1: Sync data from GCS to Modal Volume (run once)
# ============================================================================
@app.function(
    cpu=16.0,
    timeout=7200,
    secrets=[
        modal.Secret.from_name("gcs-secret"),
        modal.Secret.from_name("gcp-project-secret"),
    ],
    volumes={
        "/data": modal.Volume.from_name("training-data", create_if_missing=True),
    },
)
def sync_data_from_gcs():
    """
    One-time sync of data from GCS to Modal Volume
    """
    import os
    import json
    from pathlib import Path
    from google.cloud import storage
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry

    # --------------------------
    # GCS credential setup
    # --------------------------
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        creds_path = "/tmp/gcs_credentials.json"
        try:
            creds_data = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            with open(creds_path, "w") as f:
                json.dump(creds_data, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            logger.info("✓ GCS credentials configured")
        except Exception as e:
            logger.error(f"❌ Error setting up GCS credentials: {e}")
            raise

    bucket_name = "apcomp215-datasets"
    gcs_prefix = "dataset_v1/"
    data_dir = "/data/dataset_v2"
    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Source: gs://{bucket_name}/{gcs_prefix}")
    logger.info(f"Target: {data_dir}")
    logger.info("Starting parallel transfer (128 workers)...")

    # --------------------------
    # Initialize GCS client
    # --------------------------
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # ----------------------------------------------------
    # Increase connection pool size (POOL_SIZE = 128)
    # ----------------------------------------------------
    POOL_SIZE = 128

    retry = Retry(
        total=3,
        backoff_factor=0.2,
        status_forcelist=(500, 502, 503, 504),
    )

    adapter = HTTPAdapter(
        pool_connections=POOL_SIZE,
        pool_maxsize=POOL_SIZE,
        max_retries=retry,
        pool_block=True,  # avoid discarding connection warnings
    )

    # Mount adapter on primary GCS HTTP session
    try:
        client._http.mount("https://", adapter)
        logger.info(f"✓ Increased HTTP pool size to {POOL_SIZE}")
    except Exception as e:
        logger.warning(f"Could not mount adapter on client._http: {e}")

    # Mount adapter on internal auth session (AuthorizedSession)
    try:
        auth_sess = client._http._auth_request.session
        auth_sess.mount("https://", adapter)
        logger.info("✓ Mounted adapter on internal auth session")
    except Exception:
        pass

    # --------------------------
    # List all blobs
    # --------------------------
    logger.info("Scanning GCS bucket...")
    blobs = [b for b in bucket.list_blobs(prefix=gcs_prefix) if not b.name.endswith("/")]
    logger.info(f"Found {len(blobs):,} files to download")

    # --------------------------
    # Parallel download function
    # --------------------------
    def download_blob(blob):
        relative_path = blob.name[len(gcs_prefix) :]
        if not relative_path:
            return 0

        local_path = os.path.join(data_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Skip if file exists & size matches
        if os.path.exists(local_path) and os.path.getsize(local_path) == blob.size:
            return 0

        blob.download_to_filename(local_path)
        return blob.size

    # --------------------------
    # Parallel download execution
    # --------------------------
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(download_blob, blob): blob for blob in blobs}
        with tqdm(total=len(blobs), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    size = future.result()
                    total_bytes += size
                    pbar.update(1)
                except Exception as e:
                    logger.warning(f"Failed to download {futures[future].name}: {e}")

    # --------------------------
    # Verification & commit
    # --------------------------
    logger.info("Verifying transferred data...")
    file_count = sum(1 for _ in Path(data_dir).rglob("*") if _.is_file())
    total_size_gb = total_bytes / (1024**3)

    logger.info("=" * 70)
    logger.info("✓ SYNC COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Files transferred: {file_count:,}")
    logger.info(f"  Total size: {total_size_gb:.2f} GB")
    logger.info(f"  Location: {data_dir}")

    logger.info("Committing volume to persist data...")
    from modal import Volume

    volume = Volume.from_name("training-data")
    volume.commit()
    logger.info("✓ Volume committed - data persisted!")

    return f"Data sync successful: {file_count:,} files, {total_size_gb:.2f} GB"


# ============================================================================
# STEP 2: Training with volume data (fast local I/O)
# ============================================================================
@app.function(
    gpu="A100-80GB",
    cpu=20.0,
    timeout=14400,
    region="us-east",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/checkpoints": modal.Volume.from_name("training-checkpoints", create_if_missing=True),
        "/data": modal.Volume.from_name("training-data", create_if_missing=True),
    },
)
def train_with_volume(config_path: str = "configs/modal_template.yaml"):
    """
    Training function using pre-cached data from Modal Volume.
    Data is read from local SSD storage (fast!).
    """
    import os
    import sys

    # Suppress HuggingFace tokenizers parallelism warning (safe to disable with multiprocessing)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set up Python path FIRST, before ANY other imports
    src_path = "/app/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    ml_workflow_path = "/app/src/ml_workflow"

    # Verify package structure exists
    if not os.path.exists(ml_workflow_path):
        raise RuntimeError(f"Package not found: {ml_workflow_path}\nContents of /app/src: {os.listdir('/app/src')}")
    if not os.path.exists(f"{ml_workflow_path}/__init__.py"):
        raise RuntimeError(f"Package __init__.py not found: {ml_workflow_path}/__init__.py")

    # Set working directory for config file resolution
    os.chdir(ml_workflow_path)

    # Ensure checkpoint directory exists and is writable
    checkpoint_dir = "/checkpoints"
    try:
        os.makedirs(checkpoint_dir, mode=0o755, exist_ok=True)
        test_file = os.path.join(checkpoint_dir, ".modal_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info("Checkpoint directory %s is writable", checkpoint_dir)
    except (OSError, IOError) as e:
        logger.warning("Cannot write to %s: %s", checkpoint_dir, e)

    # Verify data volume is mounted and populated
    data_dir = "/data/dataset_v2"
    if os.path.exists(data_dir):
        file_count = len([f for f in Path(data_dir).rglob("*") if f.is_file()])
        logger.info("=" * 70)
        logger.info("✓ Data volume mounted and ready!")
        logger.info(f"  Files: {file_count:,}")
        logger.info(f"  Location: {data_dir}")
        logger.info("=" * 70)
    else:
        logger.error("=" * 70)
        logger.error(f"❌ Data directory not found: {data_dir}")
        logger.error("=" * 70)
        logger.error("You need to sync data to the volume first!")
        logger.error("Run: modal run modal_training_volume.py::sync_data_from_gcs")
        raise RuntimeError("Data volume not populated. Run sync_data_from_gcs first")

    # Login to wandb
    import wandb

    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    else:
        logger.warning("WANDB_API_KEY not found. Wandb logging may fail.")

    # Import using absolute import - this should work now that path is set
    # The package structure is: /app/src/ml_workflow/
    # With /app/src in sys.path, "ml_workflow" is importable as a package
    try:
        from ml_workflow.main import initialize_model
    except ImportError:
        logger.error("Import error details:")
        logger.error("  sys.path: %s", sys.path[:5])
        logger.error("  cwd: %s", os.getcwd())
        logger.error("  ml_workflow exists: %s", os.path.exists(ml_workflow_path))
        logger.error("  ml_workflow/__init__.py exists: %s", os.path.exists(f"{ml_workflow_path}/__init__.py"))
        if os.path.exists(ml_workflow_path):
            logger.error("  ml_workflow contents: %s", os.listdir(ml_workflow_path)[:10])
        raise

    # Resolve config path
    if not config_path.startswith("/"):
        config_path = os.path.join(ml_workflow_path, config_path)

    logger.info("Starting training with volume data...")
    logger.info("  Config: %s", config_path)
    logger.info("  Working dir: %s", os.getcwd())
    logger.info("  Python path: %s", sys.path[0])

    return_dict = initialize_model(config_path)
    trainer = return_dict["trainer"]
    trainer.train()

    return "Training completed successfully!"


# ============================================================================
# ENTRYPOINT
# ============================================================================


@app.local_entrypoint()
def main(config_path: str = "configs/modal_template.yaml"):
    """Local entrypoint to run training on Modal with cached data volume"""
    result = train_with_volume.remote(config_path=config_path)
    logger.info(f"✓ {result}")
