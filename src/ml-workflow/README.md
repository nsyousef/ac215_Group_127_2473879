Modular PyTorch dataloader supporting both **GCS** and **local storage**.

## Structure

```
ml-workflow/
├── config.py          # Configuration with env variable support
├── utils.py           # Transforms, utilities & data analysis
├── dataloader.py      # ImageDataset & create_dataloaders
└── requirements.txt   # Dependencies
```

## Quick Start

### Basic Usage (Local Files)

```python
from utils import load_metadata
from dataloader import create_dataloaders

# Load metadata (filters classes with < 10 samples by default)
metadata = load_metadata("./data/metadata.csv", min_samples=10)

# Create train/test loaders (computes mean/std from all training data)
# By default: 80% train, 20% test (test_size=0.2, val_size=None)
train_loader, val_loader, test_loader, info = create_dataloaders(
    metadata_df=metadata,
    use_local=True,
    batch_size=32,
    num_workers=4,
    compute_stats=True  # Computes mean/std from ALL training images
)

# Note: val_loader is None by default (2-way train/test split)
print(f"Classes: {info['num_classes']}")
print(f"Train size: {info['train_size']}")
print(f"Test size: {info['test_size']}")
print(f"Failed images: {len(info['failed_images']['train'])}")
```

### GCS Storage

```python
from utils import load_metadata
from dataloader import create_dataloaders
from config import GCS_BUCKET_NAME, GCS_METADATA_PATH, GCS_IMAGE_PREFIX

# Load from GCS
gcs_path = f"gs://{GCS_BUCKET_NAME}/{GCS_METADATA_PATH}"
metadata = load_metadata(gcs_path, min_samples=10)

# Create dataloaders
train_loader, val_loader, test_loader, info = create_dataloaders(
    metadata_df=metadata,
    img_prefix=GCS_IMAGE_PREFIX,
    batch_size=32,
    num_workers=4
)
```

### Split (Train/Val/Test)

For rigorous evaluation with hyperparameter tuning:

```python
# Create train/val/test split with val_size parameter
# Train: 70%, Val: 15% (for tuning), Test: 15% (for final evaluation)
train_loader, val_loader, test_loader, info = create_dataloaders(
    metadata_df=metadata,
    use_local=True,
    val_size=0.15,      # 15% for validation (hyperparameter tuning)
    test_size=0.15,     # 15% for test (final evaluation)
    batch_size=32
)

# Now test_loader is available for final evaluation
print(f"Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")
```
