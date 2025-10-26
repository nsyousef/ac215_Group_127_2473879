# ML Workflow for Skin Disease Classification

A configurable machine learning pipeline for skin disease classification built for APCOMP215.

## 📁 Directory Structure

```
ml-workflow/
├── main.py                    # Main entry point for training
├── eval.py                    # Model evaluation script
├── constants.py              # Project constants and configurations
├── utils.py                  # Utilities, checkpointing & data analysis
├── io_utils.py              # Input/output utilities
├── configs/
│   └── TEMPLATE.yaml         # Configuration template
├── dataloader/
│   ├── dataloader.py         # ImageDataset & create_dataloaders
│   ├── transform_utils.py    # Image transformations
│   └── embedding_utils.py    # Text embedding utilities
├── model/
│   ├── utils.py             # Model utilities
│   ├── classifier/
│   │   └── classifier.py    # Classification head
│   └── vision/
│       ├── cnn.py          # CNN-based models
│       └── vit.py          # Vision Transformer models
├── train/
│   ├── train.py            # Training class
│   └── train_utils.py      # Training utilities
└── requirements.txt         # Dependencies
```

## Quick Start

### 1. Training a Model

Set your parameters in the config file and run:

```bash
python main.py --config path/to/your/config.yaml
```

### 2. Evaluating a Model

Enter the checkpoint path in the config and run:

```bash
python eval.py --config configs/config.yaml --plot-dir ./my_plots
```

### 3. Using in Jupyter Notebooks

```python
from main import initialize_model

# Initialize model and data
return_dict = initialize_model('configs/config.yaml')
trainer = return_dict['trainer']
model = return_dict['vision_model']
test_loader = return_dict['test_loader']

# Train the model or evaluate as required
trainer.train()
```

## ⚙️ Configuration

The workflow is fully configurable via YAML files in the `configs/` directory.

### Data Source Configuration

| Parameter | Description |
|-----------|-------------|
| `use_local` | Whether to load data from local storage instead of cloud or remote sources |
| `metadata_path` | Path to the metadata CSV that defines image paths, labels, and text descriptions (only used if `use_local` is True) |
| `img_prefix` | Directory for image files |
| `min_samples_per_label` | Minimum number of images required for a label to be included |
| `datasets` | List of datasets to include (e.g., Fitzpatrick17k, DDI, ISIC) |
| `has_text` | If `True`, only include data with text descriptions; if `False`, only include data without text descriptions; if `null`, do not filter on presence of text descriptions. |

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `batch_size` | Number of samples per batch |
| `num_workers` | Number of CPU threads used for data loading |
| `seed` | Random seed for reproducibility |
| `prefetch_factor` | Controls how many batches each worker preloads |
| `num_epochs` | Maximum number of training epochs |
| `patience` | Number of epochs to wait before early stopping if validation doesn't improve |
| `validation_interval` | Frequency (in epochs) to run validation |
| `n_warmup_epochs` | Number of epochs to freeze the backbone |

#### Scheduler

| Parameter | Description |
|-----------|-------------|
| `use_cosine_annealing` | Enables cosine annealing learning rate scheduler |
| `backbone_eta_min` | Minimum learning rate for backbone during cosine annealing |
| `head_eta_min` | Minimum learning rate for classification head during cosine annealing |

### Data Splits

| Parameter | Description |
|-----------|-------------|
| `test_size` | Fraction of data reserved for testing |
| `val_size` | Fraction of data reserved for validation |

### Image Configuration

| Parameter | Description |
|-----------|-------------|
| `size` | Image dimensions `[height, width]` |
| `mode` | Image color mode (e.g., RGB or grayscale) |

### Augmentation Parameters

| Parameter | Description |
|-----------|-------------|
| `brightness_jitter` | Random brightness perturbations for robustness (set to `null` to disable) |
| `contrast_jitter` | Random contrast perturbations |
| `saturation_jitter` | Random saturation perturbations |
| `hue_jitter` | Random hue perturbations |
| `rotation_degrees` | Random rotation range in degrees |
| `scale` | Random scaling range (e.g., `[0.8, 1.1]`) |
| `horizontal_flip_prob` | Probability of horizontal flipping |
| `vertical_flip_prob` | Probability of vertical flipping |
| `translate` | Translation range |
| `grayscale_prob` | Probability of converting to grayscale |

### Data Processing

| Parameter | Description |
|-----------|-------------|
| `compute_stats` | Whether to recompute dataset statistics (mean, std). If false, uses precomputed ImageNet statistics |
| `weighted_sampling` | Balances image sampling frequency inversely to class counts for class imbalance correction |

### Vision Model Configuration

| Parameter | Description |
|-----------|-------------|
| `name` | Model backbone type. For CNN: ("resnet50", "resnet101", "densenet121", "efficientnet_b0", "efficientnet_b4", "vgg16") or ViT: ("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32") |
| `pretrained` | Whether to load pretrained weights |
| `img_size` | Input image size as tuple (default: (224, 224)) |
| `pooling_type` | For CNN models: Type of feature pooling ('avg', 'max', 'concat') |
| `unfreeze_layers` | Number of layers to unfreeze from end. For CNNs: backbone layers, for ViTs: transformer blocks |

### Classifier Configuration

| Parameter | Description |
|-----------|-------------|
| `num_classes` | Number of output classes (set to `null` for auto-inference based on data) |
| `hidden_sizes` | Sizes of hidden layers in classification head |
| `activation` | Non-linear activation function (e.g., "relu") |
| `dropout_rate` | Fraction of neurons dropped during training for regularization |
| `label_smoothing` | Smooths target labels to reduce overconfidence |
| `loss_fn` | Type of loss function (e.g., "cross_entropy", "focal") |
| `sample_weights` | Class weights for handling imbalanced datasets |

### Text Embedding Configuration

| Parameter | Description |
|-----------|-------------|
| `model_name` | Name of text embedding model (e.g., "pubmedbert", "qwen") |
| `batch_size` | Batch size for text encoding |
| `max_length` | Maximum sequence length for text input |
| `pooling_type` | Strategy for pooling text embeddings ('mean', 'cls', 'last_token') |
| `qwen_instr` | Instructions for QWEN models to generate task-specific embeddings |
| `pre_existing_path` | Path to pre-computed embeddings (optional)

### Optimizer Configuration

Separate configurations can be set for backbone and head optimizers:

| Parameter | Description |
|-----------|-------------|
| `name` | Optimizer algorithm (e.g., "adam", "adamw") |
| `learning_rate` | Initial learning rate (separate values can be used for backbone and head) |
| `weight_decay` | L2 regularization coefficient |
| `momentum` | Momentum term (used in SGD) |
| `betas` | Exponential decay rates for first and second moment estimates (Adam-family) |
| `eps` | Small constant added for numerical stability |

### Output and Checkpointing

| Parameter | Description |
|-----------|-------------|
| `save_dir` | Directory to save model checkpoints |
| `log_dir` | Directory to store logs and metrics |
| `experiment_name` | Label for the experiment (used for naming files) |
| `save_frequency` | Frequency (in epochs) for saving checkpoints |
| `keep_last` | Number of most recent checkpoints to keep |
| `load_from` | Path to a pretrained checkpoint to resume from |

### Example Checkpoint Configuration

The model automatically keeps the checkpoint with the best validation accuracy. You can continue training from a checkpoint by specifying the path:

```yaml
checkpoint:
  save_frequency: 5    # Save every 5 epochs
  keep_last: 10        # Keep last 10 checkpoints
  load_from: "./checkpoints/best_model.pth"  # Resume from checkpoint
```

## 📋 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
