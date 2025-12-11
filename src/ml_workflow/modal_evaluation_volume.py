"""
Modal Evaluation with Volume Cache

This script evaluates a trained model on the test set using pre-loaded data from Modal Volume.

Usage:
  modal run modal_evaluation_volume.py --config-path configs/modal_template.yaml
  modal run modal_evaluation_volume.py --config-path configs/modal_template.yaml --weights-path /checkpoints/model.pth
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
    # Exclude large/unnecessary directories to reduce upload size
    .add_local_dir(
        str(SRC_DIR),
        "/app/src",
        ignore=[
            "**/__pycache__/**",
            "**/node_modules/**",
            "**/out/**",
            "**/.git/**",
            "**/wandb/**",
            "**/logs/**",
            "**/.pytest_cache/**",
            "**/.ruff_cache/**",
            "**/dist/**",
            "**/build/**",
            "**/*.pyc",
            "**/.DS_Store",
            "**/frontend-react/**",  # Exclude entire frontend (not needed for evaluation)
            "**/cv-analysis/**",  # Exclude if not needed
        ],
    )
)

app = modal.App("skin-disease-evaluation-volume", image=image)


# ============================================================================
# Evaluation function with volume data (fast local I/O)
# ============================================================================
@app.function(
    gpu="H200",
    cpu=20.0,
    timeout=50000,
    region="us-east",
    volumes={
        "/checkpoints": modal.Volume.from_name("training-checkpoints", create_if_missing=True),
        "/data": modal.Volume.from_name("training-data", create_if_missing=True),
    },
)
def evaluate_with_volume(config_path: str = "configs/modal_template.yaml", weights_path: str = None):
    """
    Evaluation function using pre-cached data from Modal Volume.
    Loads a trained model and evaluates it on the test set.
    Returns the test accuracy as a float.

    Args:
        config_path: Path to the config YAML file
        weights_path: Optional path to model weights/checkpoint. If provided, this will
                     override the checkpoint.load_from setting in the config.
    """
    import os
    import sys
    import yaml
    import tempfile

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

    # Load and modify config if weights_path is provided
    temp_config_path = None
    if weights_path is not None:
        logger.info("=" * 70)
        logger.info("Loading config and injecting weights path...")
        logger.info(f"  Original config: {config_path}")
        logger.info(f"  Weights path: {weights_path}")
        logger.info("=" * 70)

        # Load the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Ensure checkpoint section exists
        if "checkpoint" not in config:
            config["checkpoint"] = {}

        # Set the load_from path
        config["checkpoint"]["load_from"] = weights_path
        logger.info(f"✓ Updated checkpoint.load_from to: {weights_path}")

        # Write modified config to a temporary file
        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=ml_workflow_path)
        yaml.dump(config, temp_config, default_flow_style=False)
        temp_config.close()
        temp_config_path = temp_config.name
        config_path = temp_config_path
        logger.info(f"✓ Created temporary config: {config_path}")

    logger.info("Starting evaluation with volume data...")
    logger.info("  Config: %s", config_path)
    logger.info("  Working dir: %s", os.getcwd())
    logger.info("  Python path: %s", sys.path[0])

    try:
        # Initialize model (this will load the checkpoint if specified in config)
        return_dict = initialize_model(config_path)
        trainer = return_dict["trainer"]

        # Evaluate on test set
        logger.info("=" * 70)
        logger.info("Running evaluation on test set...")
        logger.info("=" * 70)

        # Use the evaluate method if available, otherwise use test method
        test_results = trainer.test()
        accuracy = test_results["test_accuracy"]

        logger.info("=" * 70)
        logger.info("✓ EVALUATION COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"  Test Accuracy: {accuracy:.2f}%")
        logger.info("=" * 70)

        return accuracy
    finally:
        # Clean up temporary config file if it was created
        if temp_config_path is not None and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            logger.info(f"✓ Cleaned up temporary config: {temp_config_path}")


# ============================================================================
# ENTRYPOINT
# ============================================================================


@app.local_entrypoint()
def main(config_path: str = "configs/modal_template.yaml", weights_path: str = None):
    """
    Local entrypoint to run evaluation on Modal with cached data volume

    Args:
        config_path: Path to the config YAML file
        weights_path: Optional path to model weights/checkpoint. If provided, this will
                     override the checkpoint.load_from setting in the config.
    """
    accuracy = evaluate_with_volume.remote(config_path=config_path, weights_path=weights_path)
    logger.info(f"✓ Evaluation completed! Test Accuracy: {accuracy:.2f}%")
    return accuracy
