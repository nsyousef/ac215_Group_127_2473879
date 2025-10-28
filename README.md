# Milestone 3

# Skin Disease Classification - AC215 Group 127.2473879

This project combines multiple datasets (Fitzpatrick17k, DDI, ISIC, SCIN, SkinCap) into a harmonized dataset and provides a complete ML pipeline for training skin disease classification models.

## Project Overview

This system processes raw skin disease datasets from multiple sources, harmonizes their labels, and provides a complete machine learning workflow for training classification models. The project includes data processing, model training, and evaluation.

## File Structure

```
secrets/
ac215_Group_127_2473879/
├── docs/                          # Project documentation
│   ├── gcp/                       # Google Cloud Platform screenshots (running VMs and bucket structure)
│   ├── mockup/                    # UI mockups
│   └── decks/                     # Pitch deck for Milestone 3
├── eda/                           # Exploratory Data Analysis
│   ├── ddi_eda.ipynb              # DDI dataset analysis
│   ├── derm1m.ipynb               # Derm1M dataset analysis
│   ├── isic_eda.ipynb             # ISIC dataset analysis
│   ├── skincap_eda.ipynb          # SkinCap dataset analysis
│   ├── scin.ipynb                 # SCIN dataset analysis
│   └── fitzpatrick/               # Fitzpatrick dataset analysis
└── src/                           # Source code (microservices architecture)
    ├── data-processor/            # Data processing and harmonization
    │   ├── data-processor.log     # Data processing logs
    │   ├── data-processor.sh      # Runs the whole data processing pipeline
    │   └── processor_derm1m.py    # Derm1M processing   
    └── ml_workflow/               # Machine learning training pipeline
        ├── dataloader/            # Data loading utilities
        │   ├── dataloader.py      # Main dataloader implementation
        │   ├── embedding_utils.py # Text embedding utilities
        │   ├── transform_utils.py # Code for data augmentation utilities
        │   └── dataloader.ipynb   # Dataloader notebook
        ├── model/                 # Model architectures
        │   ├── classifier/        # Classifier models
        │   │   └── classifier.py  # Classifier implementation
        │   ├── vision/            # Vision models
        │   │   ├── cnn.py         # CNN architecture
        │   │   └── vit.py         # Vision Transformer architecture
        │   └── utils.py           # Model utilities
        ├── text_embeddings.ipynb  # Text embeddings analysis
        ├── test_dataloader.py     # Dataloader testing
        ├── io_utils.py            # Input/output utilities
        └── logs/                  # Training logs
```

Note: /secrets folder should be outside of ac215_Group_127_2473879.

## Microservices Description

Each folder in `src` represents a microservice with its own Docker container:

- **data-processor**: Converts raw datasets into standardized format and conducts label harmonization
- **ml_workflow**: Configurable machine learning pipeline for skin disease classification

For detailed information about each service, please refer to the individual README files in each folder.

# Milestone 3 Deliverables

1. Derm1M EDA and preprocessing (eda/derm1m.ipynb and data-processor/processor_derm1m.py)
2. Updated dataloader that includes text embeddings (dataloader/)
3. Vision transformer model (model/vision/vit.py)
4. Weights and biases logging (train/train.py)