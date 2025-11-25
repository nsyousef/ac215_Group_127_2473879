"""True constants for the dataloader pipeline"""

# GCS Configuration (fixed for this project)
GCS_BUCKET_NAME = "apcomp215-datasets"
GCS_METADATA_PATH = f"gs://apcomp215-datasets/dataset_v1/metadata_all_harmonized.csv"
GCS_IMAGE_PREFIX = f"gs://apcomp215-datasets/dataset_v1/imgs"

# Local Storage Configuration (for testing/development)
LOCAL_DATA_DIR = '../../data'
LOCAL_METADATA_PATH = '../../data/metadata_all_harmonized.csv'

# Image Configuration (typically fixed)
DEFAULT_IMAGE_MODE = "RGB"
DEFAULT_IMAGE_SIZE = (224, 224)

# Column names (fixed for this project)
IMG_COL = "filename"
LABEL_COL = "label"
TEXT_DESC_COL = "text_desc"
IMG_ID_COL = "image_id"
EMBEDDING_COL = "embedding"
ORIG_FILENAME_COL = 'orig_filename'

MAX_RETRIES = 3

# Embedding models
MODELS = {
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'biosyn': 'dmis-lab/biosyn-sapbert-bc5cdr-disease',
    'sapbert': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
    'qwen': 'Qwen/Qwen3-Embedding-8B'
}
