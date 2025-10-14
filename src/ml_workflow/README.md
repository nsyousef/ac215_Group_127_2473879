Complete PyTorch ML workflow for skin disease classification with **transfer learning**, **checkpointing**, and **evaluation**.

## Structure

```
ml-workflow/
├── main.py                    # Main entry point for training
├── eval.py                    # Model evaluation script
├── constants.py               # Project constants
├── utils.py                   # Utilities, checkpointing & data analysis
├── configs/
│   └── config.yaml           # Configuration file
├── dataloader/
│   ├── dataloader.py         # ImageDataset & create_dataloaders
│   └── transform_utils.py    # Image transformations
├── model/
│   ├── imagenet.py           # Transfer learning model
│   └── utils.py              # Model utilities (MLP)
├── train/
│   └── train.py              # Training class
└── requirements.txt          # Dependencies
```

## Quick Start

### 1. Training a Model

```bash
# Train with default configuration
python main.py --config configs/config.yaml

# Train with custom configuration
python main.py --config path/to/your/config.yaml
```

### 2. Evaluating a Model

```bash
# Evaluate with plots
python eval.py --config configs/config.yaml

# Evaluate without plots
python eval.py --config configs/config.yaml --no-plots

# Custom plot directory
python eval.py --config configs/config.yaml --plot-dir ./my_plots
```

### 3. Using in Jupyter Notebooks

```python
from main import initialize_model

# Initialize model and data
return_dict = initialize_model('configs/config.yaml')
trainer = return_dict['trainer']
model = return_dict['model']
test_loader = return_dict['test_loader']

# Train the model
trainer.train()

# Or just load a pre-trained model for inference
# (set checkpoint.load_from in config.yaml)
```

## Configuration

The workflow is fully configurable via `configs/config.yaml`:

### Key Configuration Sections:

```yaml
# Data source
data:
  use_local: true
  metadata_path: "../../data/metadata_all_harmonized.csv"
  img_prefix: "../../data"

# Model architecture
model:
  name: "resnet50"  # resnet50, resnet101, densenet121, efficientnet_b0, vgg16
  pretrained: true
  hidden_sizes: [512, 256]
  activation: "relu"

# Training parameters
training:
  batch_size: 32
  num_epochs: 100
  patience: 10

# Checkpointing
checkpoint:
  save_frequency: 10
  keep_last: 5
  load_from: null  # Path to checkpoint to resume from
```

## Features

### ✅ **Complete ML Pipeline**
- **Data Loading**: GCS and local storage support
- **Transfer Learning**: ImageNet pretrained models
- **Training**: Full training/validation/test loops
- **Checkpointing**: Save/load with resume capability
- **Evaluation**: Comprehensive metrics and visualizations

### ✅ **Model Architectures**
- ResNet50, ResNet101
- DenseNet121
- EfficientNet-B0
- VGG16

### ✅ **Data Augmentation**
- Color jittering (brightness, contrast, saturation, hue)
- Geometric transforms (rotation, translation, scaling)
- Random flips and grayscale conversion
- Configurable augmentation parameters

### ✅ **Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- Per-class performance analysis
- Confusion matrix visualization
- Class distribution plots
- Overall performance summary

### ✅ **Device Support**
- Automatic CUDA detection
- CPU fallback
- Device-agnostic code

## Advanced Usage

### Custom Model Configuration

```yaml
model:
  name: "efficientnet_b0"
  pretrained: true
  hidden_sizes: [1024, 512, 256]  # Deeper classifier
  activation: "gelu"
  dropout_rate: 0.3
```

### Advanced Augmentation

```yaml
augmentation:
  brightness_jitter: 0.2
  contrast_jitter: 0.2
  rotation_degrees: 30
  translate: [0.2, 0.2]
  scale: [0.8, 1.2]
  grayscale_prob: 0.2
```

### Checkpoint Management

```yaml
checkpoint:
  save_frequency: 5    # Save every 5 epochs
  keep_last: 10        # Keep last 10 checkpoints
  load_from: "./checkpoints/best_model.pth"  # Resume from checkpoint
```
