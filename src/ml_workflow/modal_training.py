import modal

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
    ])
    .apt_install(["git"])
    # Add your local code directly to the image
    .add_local_dir(".", "/app")  # This replaces Mount.from_local_dir()
)

app = modal.App("skin-disease-training", image=image)

@app.function(
    gpu=modal.gpu.A10G(),
    timeout=86400,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("gcs-secret"),
    ],
    volumes={
        "/checkpoints": modal.Volume.from_name("training-checkpoints", create_if_missing=True),
        "/gcs-data": modal.CloudBucketMount(
            bucket_name="derma-datasets-2",
            secret=modal.Secret.from_name("gcs-secret")
        )
    }
)
def train_with_gcs(config_path: str = "configs/modal_template.yaml"):
    import os
    import sys
    
    os.chdir("/app")
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    
    import wandb
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    
    from main import initialize_model
    
    return_dict = initialize_model(config_path)
    trainer = return_dict['trainer']
    trainer.train()
    
    return "Training completed successfully!"

@app.local_entrypoint()
def main(config_path: str = "configs/modal_template.yaml"):
    result = train_with_gcs.remote(config_path=config_path)
    print(result)