# Milestone 4: Documentation

## Application Design Document (including solution and technical architecture)

<!-- TODO: Fill in application design, solution overview, and technical architecture details -->

## Data Versioning documentation (methodology, justification, and usage instructions)

### Methodology: Snapshot-Based Versioning

We employ a **snapshot-based data versioning strategy** for our skin disease classification dataset. This approach is well-suited for our use case because:

- **Static Dataset**: Our data consists of curated medical images from established datasets (Fitzpatrick17k, DDI, ISIC, SCIN, SkinCap, Derm1M) that do not change once processed
- **Reproducibility**: Each snapshot represents a complete, immutable version of the processed dataset
- **Simplicity**: Avoids the complexity of delta-based or streaming data versioning systems
- **Cost-Effective**: Leverages Google Cloud Storage for efficient snapshot storage

### Storage Structure

Our versioned datasets are stored in Google Cloud Storage with the following organization:

```
gs://apcomp215-datasets/
├── dataset_v1/                          # First production dataset snapshot
│   ├── imgs/                            # Harmonized image files
│   │   ├── fitzpatrick17k_<id>.<ext>
│   │   ├── ddi_<id>.<ext>
│   │   ├── isic_<id>.<ext>
│   │   ├── scin_<id>.<ext>
│   │   ├── skincap_<id>.<ext>
│   │   └── derm1m_<id>.<ext>
│   ├── emb/                             # Pre-computed text embeddings
│   │   ├── pubmedbert_embeddings.parquet
│   │   └── sapbert_embeddings.parquet
│   └── metadata_all_harmonized.csv      # Unified metadata with harmonized labels
│
├── dataset_v2/                          # Future snapshot (planned)
    └── (improved text descriptions and further filtering)
```
### Experiment-Level Data Tracking

While dataset snapshots remain immutable, **experiment configuration files** provide granular control over which data is used in each training run. This two-tier approach allows us to:

1. **Maintain stable base datasets** (dataset_v1, dataset_v2, etc.)
2. **Flexibly filter data per experiment** through YAML configuration files

Each training experiment's configuration file specifies:

```yaml
data:
  metadata_path: "gs://apcomp215-datasets/dataset_v1/metadata_all_harmonized.csv"
  img_prefix: "gs://apcomp215-datasets/dataset_v1/imgs"

  # Data filtering parameters (tracked per experiment)
  min_samples_per_label: 50           # Minimum images required per class
  datasets: ["fitzpatrick17k", "ddi", "isic"]  # Which source datasets to include
  derm1m_sources: ["fitzpatrick17k"]  # Specific Derm1M sources (if applicable)
  has_text: true                       # Include only samples with text descriptions
  data_fraction: 1.0                   # Fraction of data to use (for ablation studies)

  # Split configuration
  test_size: 0.15
  val_size: 0.15
  seed: 42                             # Ensures reproducible splits
```

**Benefits of this approach:**

- **Reproducibility**: Given a config file, we can exactly reproduce the data used in any experiment
- **Traceability**: Configs are version-controlled in Git alongside code
- **Flexibility**: Easy to run ablation studies (e.g., impact of text descriptions, dataset composition)
- **No Data Duplication**: Multiple experiments share the same base snapshot, filtered dynamically

### Version Evolution: dataset_v1 → dataset_v2

**dataset_v1** (Current Production)
- Contains ~190K images from 6 major dermatology datasets
- Harmonized labels (reduced from ~500 to ~100 disease categories)
- Pre-computed text embeddings (PubMedBERT, SapBERT)

**dataset_v2** (Planned)
- Enhanced text description filtering to remove leakage
- Additional quality filtering for images and metadata

**Migration Path**:
```yaml
# Old experiment (dataset_v1)
data:
  metadata_path: "gs://apcomp215-datasets/dataset_v1/metadata_all_harmonized.csv"

# New experiment (dataset_v2)
data:
  metadata_path: "gs://apcomp215-datasets/dataset_v2/metadata_all_harmonized.csv"
```

We intentionally chose a **simple snapshot-based approach** over more complex data versioning systems (e.g., DVC, Delta Lake, MLflow Data) for the above reasons.

**When we would need more complex versioning:**
- If we had continuous data ingestion from clinical partners
- If we needed to track fine-grained changes to individual images
- If we had multiple teams making concurrent modifications to the dataset

For our current scale (~190K images) and workflow (academic research project), snapshot versioning provides the optimal balance of simplicity and reproducibility.

## Model Training/Fine-Tuning summary (training process, results, and deployment implications)

<!-- TODO: Fill in model training process, results, and deployment implications -->
