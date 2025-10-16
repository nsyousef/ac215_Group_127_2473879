# Milestone 2

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
│   └── mockup/                    # UI mockups
├── eda/                           # Exploratory Data Analysis
│   ├── ddi_eda.ipynb              # DDI dataset analysis
│   ├── isic_eda.ipynb             # ISIC dataset analysis
│   ├── skincap_eda.ipynb          # SkinCap dataset analysis
│   ├── scin.ipynb                 # SCIN dataset analysis
│   └── fitzpatrick/               # Fitzpatrick dataset analysis
└── src/                           # Source code (microservices architecture)
    ├── data-processor/            # Data processing and harmonization
    │   ├── data-processor.log     # Data processing logs
    │   ├── data-processor.sh      # Runs the whole data processing pipeline
    └── ml_workflow/               # Machine learning training pipeline
        └── logs/                  # Training logs
```

Note: /secrets folder should be outside of ac215_Group_127_2473879.

## Microservices Description

Each folder in `src` represents a microservice with its own Docker container:

- **data-processor**: Converts raw datasets into standardized format and conducts label harmonization
- **ml_workflow**: Configurable machine learning pipeline for skin disease classification

For detailed information about each service, please refer to the individual README files in each folder.

# Milestone 2 Deliverables

1. Virtual Environment Setup - screenshot of running VMs on the GCP (`docs/gcp`)
2. End-To-End Containerized Pipeline (`src/data-processor/data-processor.sh`)
3. Vision/ML workflow (`src/ml_workflow`)
4. Application Mockup (`docs/mockup.md`)