# Milestone 4

# Skin Disease Classification - AC215 Group 127.2473879

This project combines multiple datasets (Fitzpatrick17k, DDI, ISIC, SCIN, Derm1M) into a harmonized dataset and provides a complete ML pipeline for training skin disease classification models.

## Project Overview

This system processes raw skin disease datasets from multiple sources, harmonizes their labels, and provides a complete machine learning workflow for training classification models. The project includes data processing, model training, and evaluation.

## File Structure

```
secrets/
ac215_Group_127_2473879/
├── docs/                          # Project documentation
│   ├── architecture.pdf           # Application design document
│   ├── training_summary.pdf       # Model training/fine-tuning document
│   ├── milestone4.md              # Data versioning document
│   ├── gcp/                       # Google Cloud Platform screenshots
│   ├── mockup/                    # UI mockups
│   └── decks/                     # Pitch deck
├── eda/                           # Exploratory Data Analysis
│   ├── ddi_eda.ipynb              # DDI dataset analysis
│   ├── derm1m.ipynb               # Derm1M dataset analysis
│   ├── isic_eda.ipynb             # ISIC dataset analysis
│   ├── skincap_eda.ipynb          # SkinCap dataset analysis
│   ├── scin.ipynb                 # SCIN dataset analysis
│   └── fitzpatrick/               # Fitzpatrick dataset analysis
└── src/                           # Source code (microservices architecture)
    ├── data-processor/            # Data processing and harmonization
    │   ├── processor_*.py         # Dataset processors
    │   ├── data-processor.sh      # Pipeline runner
    │   └── tests/                 # Unit tests
    ├── ml_workflow/               # Machine learning training pipeline
    │   ├── dataloader/            # Data loading utilities
    │   ├── model/                 # Model architectures (CNN, ViT, classifiers)
    │   ├── configs/               # Training configurations
    │   └── logs/                  # Training logs
    ├── inference-cloud/           # Cloud inference service
    │   ├── main.py                # Inference API
    │   └── tests/                 # Unit, integration, system tests
    ├── llm/                       # LLM service for medical recommendations
    │   ├── llm.py                 # LLM implementation
    │   └── tests/                 # Unit, integration, system tests
    └── frontend-react/            # React frontend application
        ├── src/                   # React components and pages
        ├── python/                # Backend API services
        └── tests/                 # Integration tests
```

Note: /secrets folder should be outside of ac215_Group_127_2473879.

## Microservices Description

Each folder in `src` represents a microservice with its own Docker container:

- **data-processor**: Converts raw datasets into standardized format and conducts label harmonization
- **ml_workflow**: Configurable machine learning pipeline for skin disease classification
- **inference-cloud**: Cloud-based inference service for model predictions
- **llm**: LLM service for generating medical recommendations and insights
- **frontend-react**: React-based web application with Python backend API

For detailed information about each service, please refer to the individual README files in each folder.

# Milestone 4 Deliverables

1. **Application Design Document**: [`docs/architecture.pdf`](docs/architecture.pdf)
2. **Data Versioning Document**: [`docs/milestone4.md`](docs/milestone4.md)
3. **Model Training/Fine-tuning Document**: [`docs/training_summary.pdf`](docs/training_summary.pdf) and training logs in [`src/ml_workflow/logs/`](src/ml_workflow/logs/)
4. **Testing Documentation**: 
   - Unit tests: `src/data-processor/tests/`, `src/llm/tests/unit.py`, `src/inference-cloud/tests/test_unit.py`
   - Integration tests: `src/frontend-react/python/tests/integration.py`, `src/llm/tests/integration.py`, `src/inference-cloud/tests/test_integration.py`
   - System tests: `src/llm/tests/system.py`, `src/inference-cloud/tests/test_system.py`
   - Run tests using `pytest` or the respective `run_*_tests.sh` scripts in each service directory
5. **APIs and Frontend**: See [`src/frontend-react/README.md`](src/frontend-react/README.md)
6. **CI/CD Configuration**: Dockerfile.ci files in `src/data-processor/`, `src/llm/`, `src/frontend-react/`, and `src/inference-cloud/`
