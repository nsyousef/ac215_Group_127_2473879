"""True constants for the dataloader pipeline"""

# GCS Configuration (fixed for this project)
GCS_BUCKET_NAME = "derma-datasets-2"
GCS_METADATA_PATH = f"gs://derma-datasets-2/final/metadata_all_harmonized.csv"
GCS_IMAGE_PREFIX = f"gs://derma-datasets-2/final/imgs"

# Image Configuration (typically fixed)
DEFAULT_IMAGE_MODE = "RGB"

# Column names (fixed for this project)
IMG_COL = "filename"
LABEL_COL = "label"
