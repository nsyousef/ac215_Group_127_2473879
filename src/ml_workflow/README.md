## ML Workflow

Training pipeline for multimodal skin disease classification using vision + text embeddings.

### Directory Structure

```
ml_workflow/
├── main.py                    # Local training entry point
├── modal_training_volume.py   # Modal training with Volume cache (RECOMMENDED)
├── modal_evaluation_volume.py # Modal model evaluation on test set
├── modal_training_gcs.py      # Modal training with GCS (rarely used)
├── constants.py               # Project constants
├── utils.py                   # Utilities, checkpointing & data analysis
├── io_utils.py                # Input/output utilities
├── configs/
│   ├── experiments/          # Experiment configurations
│   └── modal_template.yaml   # Template for Modal training
├── dataloader/
│   ├── dataloader.py         # Dataset & dataloaders
│   ├── transform_utils.py    # Image augmentations
│   └── embedding_utils.py    # Text embedding utilities
├── model/
│   ├── classifier/
│   │   └── multimodal_classifier.py  # Multimodal classification head
│   └── vision/
│       ├── cnn.py            # CNN models (ResNet, EfficientNet, etc.)
│       └── vit.py            # Vision Transformer models
└── train/
    └── train.py              # Training logic
```

### Quick Start

#### Local Training
```bash
python main.py --config configs/your_config.yaml
```
Set `data.use_local: true` in config.

#### Modal Training (Recommended)

**Step 1: Sync data to Modal Volume (one-time)**
```bash
modal run --detach modal_training_volume.py::sync_data_from_gcs
```

**Step 2: Train**
```bash
modal run --detach modal_training_volume.py --config-path configs/modal_template.yaml
```
Set `data.use_local: true` and use paths like `/data/dataset_v1/...`.

#### Modal Evaluation
```bash
modal run modal_evaluation_volume.py --config-path configs/modal_template.yaml
```

### Configuration

All training is configured via YAML files in `configs/`. Key sections:

**Data Configuration**
- `use_local`: `true` (local/volume) or `false` (GCS)
- `dataset`: Dataset version (e.g., `dataset_v2`)
- `min_samples_per_label`: Minimum images per class
- `datasets`: List to include (`fitzpatrick17k`, `ddi`, `isic`, `derm1m`)
- `per_dataset_split_ratios`: Per-dataset train/val/test splits (optional, overrides `test_size`/`val_size`)
- `derm1m_sources`: Derm1M sources filter
- `has_text`: Filter by text presence (`true`/`false`/`null`)
- `data_fraction`: Fraction to use (0.0-1.0 or `null`)

**Training Parameters**
- `batch_size`, `num_workers`, `prefetch_factor`
- `num_epochs`, `patience`, `validation_interval`
- `n_warmup_epochs`: Freeze backbone epochs
- `use_normalization`: Use dataset normalization (recommended)
- `weighted_sampling`: Balance class frequencies
- `vision_only_pretraining.enabled`: Enable vision-only pre-training
- `vision_only_pretraining.epochs`: Epochs for vision-only pre-training
- `auxiliary_loss_schedule`: Auxiliary loss scheduling (replaces `auxiliary_loss_weight`)
  - `vision.start_weight`, `vision.end_weight`: Vision auxiliary loss weights
  - `text.start_weight`, `text.end_weight`: Text auxiliary loss weights
- `scheduler.use_cosine_annealing`: Enable cosine LR schedule
- `scheduler.vision_eta_min`, `scheduler.multimodal_classifier_eta_min`: Minimum LR values

**Data Splits**
- `test_size`, `val_size`: Train/val/test split fractions (used if `per_dataset_split_ratios` not provided)

**Image Configuration**
- `img_size`: Image dimensions `[height, width]`
- `seed`: Random seed

**Augmentation**
- `brightness_jitter`, `contrast_jitter`, `saturation_jitter`, `hue_jitter`
- `rotation_degrees`, `scale`, `translate`
- `horizontal_flip_prob`, `vertical_flip_prob`
- `grayscale_prob`

**Vision Model**
- `name`: Backbone (`resnet50`, `resnet101`, `efficientnet_b0/b1/b4`, `vit_b_16`, etc.)
- `pretrained`: Load pretrained weights
- `pooling_type`: Feature pooling (`avg`, `max`, `concat` - `concat` recommended for EfficientNet)
- `unfreeze_layers`: Number of layers to unfreeze (`-1` = unfreeze all)

**Multimodal Classifier**
- `projection_dim`: Common embedding dimension
- `use_l2_normalization`: Normalize before fusion
- `image_projection_hidden`, `text_projection_hidden`: Hidden layer sizes
- `text_projection_dropout`: Separate dropout for text projection
- `projection_dropout`, `final_dropout`: Dropout rates
- `projection_activation`, `final_activation`: Activation functions (`relu` or `gelu`)
- `fusion_strategy`: `weighted_sum` or `concat_mlp` (`concat_mlp` recommended)
- `use_auxiliary_loss`: Enable auxiliary losses
- `loss_fn`: `cross_entropy` or `focal`
- `label_smoothing`: Label smoothing value (for cross_entropy)
- `gamma`: Focusing parameter for focal loss
- `use_class_weights_from_data`: Auto class weights from data
- `class_weights`: Manual class weights

**Modality Masking**
- `mask_complete`: `null`, `image`, or `text` (mask entire modality)
- `random_mask.enabled`: Enable random masking (modality dropout)
- `random_mask.image_prob`, `random_mask.text_prob`: Masking probabilities (0.0-1.0)
- `epoch_schedule.enabled`: Enable epoch-based masking schedule

**Text Embedding**
- `model_name`: Model (`pubmedbert`, `biosyn`, `sapbert`, `qwen`)
- `batch_size`, `max_length`
- `pooling_type`: `mean`, `cls`, or `last_token`
- `qwen_instr`: Instructions for QWEN models
- `embedding_filename`: Pre-computed embeddings filename

**Optimizer** (separate for vision and classifier)
- `name`: `adam`, `adamw`, `sgd`
- `learning_rate`, `weight_decay`, `momentum`, `betas`, `eps`

**Checkpointing**
- `save_dir`, `log_dir`, `experiment_name`
- `wandb_project`: W&B project name
- `save_frequency`: Save every N epochs
- `keep_last`: Keep N most recent checkpoints
- `load_from`: Resume from checkpoint path
