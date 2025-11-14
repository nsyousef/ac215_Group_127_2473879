import modal
from pathlib import Path

# Get paths - need to mount src/ directory so ml_workflow package structure works
SRC_DIR = Path(__file__).parent.parent  # Goes up to src/
ML_WORKFLOW_DIR = Path(__file__).parent  # ml_workflow directory

# Define the image with all required dependencies AND your local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
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
        "google-cloud-storage",
        "pyarrow",
        "gcsfs",  # For GCS filesystem access
    ])
    .apt_install(["git"])
    # Mount src/ directory so ml_workflow package structure works
    # This creates /app/src/ml_workflow/ structure
    .add_local_dir(str(SRC_DIR), "/app/src")
)

app = modal.App("skin-disease-training", image=image)

@app.function(
    gpu="A100-40GB",
    timeout=7200,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("gcs-secret"),
    ],
    volumes={
        "/checkpoints": modal.Volume.from_name("training-checkpoints", create_if_missing=True)
    }
)
def train_with_gcs(config_path: str = "configs/modal_template.yaml"):
    import os
    import sys
    import json
    
    # Set up GCS authentication by writing JSON credentials to a file
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        creds_path = "/tmp/gcs_credentials.json"
        try:
            # Parse the JSON string and write to file
            creds_data = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            with open(creds_path, 'w') as f:
                json.dump(creds_data, f)
            
            # Set the environment variable that gcsfs will use
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            print(f"✓ GCS credentials written to {creds_path}")
        except json.JSONDecodeError as e:
            print(f"⚠ Error parsing GCS credentials JSON: {e}")
        except Exception as e:
            print(f"⚠ Error setting up GCS credentials: {e}")
    else:
        print("⚠ Warning: GOOGLE_APPLICATION_CREDENTIALS_JSON not found")
    
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
        test_file = os.path.join(checkpoint_dir, '.modal_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"✓ Checkpoint directory {checkpoint_dir} is writable")
    except (OSError, IOError) as e:
        print(f"⚠ Warning: Cannot write to {checkpoint_dir}: {e}")
    
    # Remove GCS mount verification since we're not mounting
    print("✓ Using direct gs:// path access (no mount needed)")
    
    # Login to wandb
    import wandb
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    else:
        print("Warning: WANDB_API_KEY not found. Wandb logging may fail.")
    
    # Import using absolute import - this should work now that path is set
    # The package structure is: /app/src/ml_workflow/
    # With /app/src in sys.path, "ml_workflow" is importable as a package
    try:
        from ml_workflow.main import initialize_model
    except ImportError as e:
        print(f"Import error details:")
        print(f"  sys.path: {sys.path[:5]}")
        print(f"  cwd: {os.getcwd()}")
        print(f"  ml_workflow exists: {os.path.exists(ml_workflow_path)}")
        print(f"  ml_workflow/__init__.py exists: {os.path.exists(f'{ml_workflow_path}/__init__.py')}")
        if os.path.exists(ml_workflow_path):
            print(f"  ml_workflow contents: {os.listdir(ml_workflow_path)[:10]}")
        raise

    # Resolve config path
    if not config_path.startswith("/"):
        config_path = os.path.join(ml_workflow_path, config_path)
    
    print(f"Starting training...")
    print(f"  Config: {config_path}")
    print(f"  Working dir: {os.getcwd()}")
    print(f"  Python path: {sys.path[0]}")
    
    return_dict = initialize_model(config_path)
    trainer = return_dict['trainer']
    trainer.train()
    
    return "Training completed successfully!"

@app.local_entrypoint()
def main(config_path: str = "configs/modal_template.yaml"):
    """
    Local entrypoint to run training on Modal
    
    Usage:
        modal run modal_training.py --config-path configs/modal_template.yaml
    """
    result = train_with_gcs.remote(config_path=config_path)
    print(result)
    