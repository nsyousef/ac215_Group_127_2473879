## ML Workflow

### Directory Structure

```
ml_workflow/
├── main.py                    # Main entry point for local training
├── modal_training_gcs.py      # Modal training with GCS CloudBucketMount
├── modal_training_volume.py   # Modal training with Volume cache (recommended)
├── constants.py               # Project constants and configurations
├── utils.py                   # Utilities, checkpointing & data analysis
├── io_utils.py                # Input/output utilities
├── configs/
│   ├── experiments/          # Experiment configuration files
│   └── modal_template.yaml   # Template for Modal training
├── dataloader/
│   ├── dataloader.py         # ImageDataset & create_dataloaders
│   ├── transform_utils.py    # Image transformations
│   └── embedding_utils.py    # Text embedding utilities
├── model/
│   ├── utils.py              # Model utilities
│   ├── classifier/
│   │   └── multimodal_classifier.py  # Multimodal classification head
│   └── vision/
│       ├── cnn.py            # CNN-based models
│       └── vit.py            # Vision Transformer models
├── train/
    └── train.py              # Training class

```

### Quick Start

The workflow supports three training methods:

#### 1. Local Training

Train on your local machine with local data storage:

```bash
python main.py --config path/to/your/config.yaml
```

**Configuration for local training:**
- Set `data.use_local: true` in your config
- Provide `data.metadata_path` and `data.img_prefix` pointing to local paths

#### 2. Modal Training with GCS

Train on Modal with direct access to GCS bucket:

```bash
modal run --detach modal_training_gcs.py --config-path configs/modal_template.yaml
```

**Configuration for GCS training:**
- Set `data.use_local: false` in your config (uses GCS paths from constants)
- Requires GCS credentials configured in Modal secrets

#### 3. Modal Training with Volume (Recommended)

**Step 1: Sync data from GCS to Modal Volume (one-time)**
```bash
modal run --detach modal_training_volume.py::sync_data_from_gcs
```

**Step 2: Train with cached data**
```bash
modal run --detach modal_training_volume.py --config-path configs/modal_template.yaml
```

**Configuration for Volume training:**
- Set `data.use_local: true` in your config
- Use paths like `/data/dataset_v1/...` (Modal Volume paths)

### Configuration

The workflow is fully configurable via YAML files in the `configs/` directory.

### Data Source Configuration

| Parameter | Description |
|-----------|-------------|
| `use_local` | Whether to load data from local storage (`true`) or GCS (`false`) |
| `dataset` | Dataset name that explicitly track the version (e.g., `dataset_v1`). Auto-constructs `metadata_path` and `img_prefix` based on this value |
| `min_samples_per_label` | Minimum number of images required for a label to be included |
| `datasets` | List of datasets to include (e.g., `fitzpatrick17k`, `ddi`, `isic`, `derm1m`) |
| `derm1m_sources` | List of derm1m sources to include (only applicable if `derm1m` is in `datasets`) |
| `has_text` | If `true`, only include data with text descriptions; if `false`, only include data without text descriptions; if `null`, do not filter on presence of text descriptions |
| `data_fraction` | Fraction of data to use (0.0-1.0, or `null` for all data) |

**Note:** The `dataset` parameter automatically constructs paths:
- **Local mode** (`use_local: true`): `/data/{dataset}/metadata_all_harmonized.csv` and `/data/{dataset}/imgs`
- **GCS mode** (`use_local: false`): `gs://apcomp215-datasets/{dataset}/metadata_all_harmonized.csv` and `gs://apcomp215-datasets/{dataset}/imgs`

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `batch_size` | Number of samples per batch |
| `num_workers` | Number of CPU threads used for data loading |
| `prefetch_factor` | Controls how many batches each worker preloads |
| `num_epochs` | Maximum number of training epochs |
| `patience` | Number of epochs to wait before early stopping if validation doesn't improve |
| `validation_interval` | Frequency (in epochs) to run validation |
| `n_warmup_epochs` | Number of epochs to freeze the backbone |
| `compute_stats` | Whether to recompute dataset statistics (mean, std). If `false`, uses precomputed ImageNet statistics |
| `weighted_sampling` | Balances image sampling frequency inversely to class counts for class imbalance correction |

#### Scheduler

| Parameter | Description |
|-----------|-------------|
| `use_cosine_annealing` | Enables cosine annealing learning rate scheduler |
| `vision_eta_min` | Minimum learning rate for backbone during cosine annealing |
| `multimodal_classifier_eta_min` | Minimum learning rate for classification head during cosine annealing |

### Data Splits

| Parameter | Description |
|-----------|-------------|
| `test_size` | Fraction of data reserved for testing |
| `val_size` | Fraction of data reserved for validation |

### Image Configuration

| Parameter | Description |
|-----------|-------------|
| `img_size` | Image dimensions `[height, width]` |
| `seed` | Random seed for reproducibility |

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
| `translate` | Translation range (set to `null` to disable) |
| `grayscale_prob` | Probability of converting to grayscale (set to `null` to disable) |

### Vision Model Configuration

| Parameter | Description |
|-----------|-------------|
| `name` | Model backbone type. For CNN: `"resnet50"`, `"resnet101"`, `"densenet121"`, `"efficientnet_b0"`, `"efficientnet_b1"`, `"efficientnet_b4"`, `"vgg16"`. For ViT: `"vit_b_16"`, `"vit_b_32"`, `"vit_l_16"`, `"vit_l_32"` |
| `pretrained` | Whether to load pretrained weights |
| `pooling_type` | For CNN models: Type of feature pooling (`'avg'`, `'max'`, `'concat'`) |
| `unfreeze_layers` | Number of layers to unfreeze from end. For CNNs: backbone layers, for ViTs: transformer blocks |

### Multimodal Classifier Configuration

| Parameter | Description |
|-----------|-------------|
| `projection_dim` | Common projection dimension for both vision and text embeddings |
| `use_l2_normalization` | Normalize embeddings before fusion |
| `image_projection_hidden` | Hidden layer sizes for vision projection (empty list = just linear layer) |
| `text_projection_hidden` | Hidden layer sizes for text projection (empty list = just linear layer) |
| `text_projection_dropout` | Dropout rate for text projection layers |
| `projection_activation` | Activation function for projection layers (e.g., `"relu"`) |
| `projection_dropout` | Dropout rate for projection layers |
| `final_hidden_sizes` | Hidden layer sizes after fusion |
| `final_activation` | Activation function for final classifier |
| `final_dropout` | Dropout rate for final classifier |
| `fusion_strategy` | Fusion strategy: `"weighted_sum"` or `"concat_mlp"` |
| `use_auxiliary_loss` | Enable auxiliary losses on individual modalities |
| `auxiliary_loss_weight` | Weight for auxiliary losses (0.0-1.0) |
| `loss_fn` | Type of loss function: `"cross_entropy"` or `"focal"` |
| `label_smoothing` | Smooths target labels to reduce overconfidence (for cross_entropy loss) |
| `use_class_weights_from_data` | Automatically inject class weights from dataloader (if `true` and no manual `class_weights` provided) |
| `class_weights` | Manual class weights for handling imbalanced datasets (optional) |

### Modality Masking Configuration

| Parameter | Description |
|-----------|-------------|
| `mask_complete` | Complete masking: `null` (no complete masking), `"image"` (mask all images), or `"text"` (mask all text) |
| `random_mask.enabled` | Enable random masking during training (modality dropout) |
| `random_mask.image_prob` | Probability of masking each image (0.0-1.0) |
| `random_mask.text_prob` | Probability of masking each text embedding (0.0-1.0) |

**Note:** The system ensures at least one modality remains unmasked for each sample.

### Text Embedding Configuration

| Parameter | Description |
|-----------|-------------|
| `model_name` | Name of text embedding model (e.g., `"pubmedbert"`, `"biosyn"`, `"sapbert"`, `"qwen"`) or path to a model |
| `batch_size` | Batch size for text encoding |
| `max_length` | Maximum sequence length for text input |
| `pooling_type` | Strategy for pooling text embeddings (`'mean'`, `'cls'`, `'last_token'`) |
| `qwen_instr` | Instructions for QWEN models to generate task-specific embeddings |
| `embedding_filename` | Filename of pre-computed embeddings (e.g., `"pubmedbert_512_cls.parquet"`). Full path is auto-constructed from `dataset` |

### Optimizer Configuration

Separate configurations can be set for vision model and multimodal classifier optimizers:

| Parameter | Description |
|-----------|-------------|
| `name` | Optimizer algorithm (e.g., `"adam"`, `"adamw"`, `"sgd"`) |
| `learning_rate` | Initial learning rate (separate values can be used for vision model and classifier) |
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
| `wandb_project` | Weights & Biases project name |
| `save_frequency` | Frequency (in epochs) for saving checkpoints |
| `keep_last` | Number of most recent checkpoints to keep |
| `load_from` | Path to a pretrained checkpoint to resume from (`null` for no loading) |

### Example Checkpoint Configuration

The model automatically keeps the checkpoint with the best validation accuracy. You can continue training from a checkpoint by specifying the path:

```yaml
checkpoint:
  save_frequency: 5    # Save every 5 epochs
  keep_last: 10        # Keep last 10 checkpoints
  load_from: "./checkpoints/best_model.pth"  # Resume from checkpoint
```
