from pathlib import Path as _Path
import warnings
from utils import logger
import pandas as pd

def file_exists(path: str) -> bool:
    """
    Return True if the given path exists locally or in Google Cloud Storage.
    - Local paths (no "gs://") are checked with pathlib and do not require GCP libs/auth.
    - GCS paths ("gs://bucket/name") are checked via google.cloud.storage and will use
      existing Google Cloud credentials (lazy-imported so local checks don't require GCP).

    NOTE: using this to check if files exist on Google Cloud will save on egress costs, compared to attempting to download the file from Google Cloud.
    """
    if not pd.isna(path):
        if path.startswith("gs://"):
            # Lazy import so local checks don't require google-cloud-storage to be installed.
            try:
                from google.cloud import storage
            except Exception:
                logger.warning("WARNING: Google cloud could not be imported. Please check that it is installed.")
                return False

            # Parse "gs://bucket/path/to/object" into bucket and object parts
            rest = path[5:]
            if not rest:
                return False
            parts = rest.split("/", 1)
            bucket_name = parts[0]
            object_path = parts[1] if len(parts) > 1 else ""

            try:
                client = storage.Client()
                if object_path == "":
                    # If no object path provided, check bucket existence.
                    bucket = client.lookup_bucket(bucket_name)
                    return bucket is not None
                else:
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(object_path)
                    return blob.exists(client=client)
            except Exception:
                return False
        else:
            return _Path(path).exists()
    else:
        return False
    
def _parse_gs_path(path: str):
    """
    Parse a GCS path of the form gs://bucket/name and return (bucket, object_path).
    Raises ValueError for invalid GCS paths.
    """
    if not path.startswith("gs://"):
        raise ValueError("Not a GCS path")
    rest = path[5:]
    if not rest:
        raise ValueError("Invalid GCS path: missing bucket")
    parts = rest.split("/", 1)
    bucket_name = parts[0]
    object_path = parts[1] if len(parts) > 1 else ""
    return bucket_name, object_path


def _get_gcs_client():
    """
    Lazy import and return a google.cloud.storage.Client instance.
    Emits a warning and raises ImportError if the library is not available.
    """
    try:
        from google.cloud import storage
    except Exception:
        warnings.warn("WARNING: Google cloud could not be imported. Please check that it is installed.")
        raise
    return storage.Client()


def save_dataframe_to_csv(path: str, df, index: bool = False, **to_csv_kwargs):
    """
    Save a pandas DataFrame to the provided path, which may be local or a GCS path (gs://...).
    - For local paths: create parent directories if needed and use DataFrame.to_csv.
    - For GCS paths: upload the CSV bytes to the bucket/object.
    to_csv_kwargs are forwarded to pandas.DataFrame.to_csv.
    """
    if path.startswith("gs://"):
        bucket_name, object_path = _parse_gs_path(path)
        if object_path == "":
            raise ValueError("GCS path must include an object name (gs://bucket/path/to/file.csv)")

        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)

        # Convert DataFrame to CSV string and upload
        csv_data = df.to_csv(index=index, **to_csv_kwargs)
        blob.upload_from_string(csv_data, content_type="text/csv")
    else:
        # Local filesystem
        p = _Path(path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=index, **to_csv_kwargs)

def save_dataframe_to_parquet(path: str, df, index: bool = False, **to_parquet_kwargs):
    """
    Save a pandas DataFrame to the provided path (local or gs://).
    - For local paths: creates parent dirs, uses DataFrame.to_parquet.
    - For GCS: uploads Parquet bytes to the bucket/object.
    to_parquet_kwargs are forwarded to pandas.DataFrame.to_parquet.
    """
    import io

    if path.startswith("gs://"):
        bucket_name, object_path = _parse_gs_path(path)
        if object_path == "":
            raise ValueError("GCS path must include an object name (gs://bucket/path/to/file.parquet)")

        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)

        # Write DataFrame to an in-memory buffer
        buf = io.BytesIO()
        df.to_parquet(buf, index=index, **to_parquet_kwargs)
        buf.seek(0)
        blob.upload_from_file(buf, content_type="application/octet-stream")
        buf.close()
    else:
        p = _Path(path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=index, **to_parquet_kwargs)


def load_csv(path: str, **read_csv_kwargs):
    """
    Load a CSV into a pandas DataFrame from a local path or a Google Cloud Storage path (gs://...).

    - Local paths are loaded with :pyfunc:`pandas.read_csv`.
    - GCS paths are downloaded via `google.cloud.storage` (lazy-imported) and then read into
      pandas using an in-memory buffer.

    Any keyword arguments are forwarded to ``pandas.read_csv``.

    Raises:
        ValueError: if a GCS path is missing the object name (e.g. only "gs://bucket").
        ImportError: if the google-cloud-storage package is required but not installed.
        RuntimeError: if downloading the GCS object fails.
    """
    if path.startswith("gs://"):
        bucket_name, object_path = _parse_gs_path(path)
        if object_path == "":
            raise ValueError("GCS path must include an object name (gs://bucket/path/to/file.csv)")

        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)

        try:
            # download as bytes and try to decode as UTF-8 for pandas
            data = blob.download_as_string()
        except Exception as e:
            raise RuntimeError(f"Failed to download GCS object '{path}': {e}")

        # Lazy import pandas and io to avoid adding heavy deps at module import time
        import io
        import pandas as pd

        # Prefer decoding to text (StringIO) but fall back to BytesIO if needed
        try:
            text = data.decode("utf-8")
            return pd.read_csv(io.StringIO(text), **read_csv_kwargs)
        except Exception:
            return pd.read_csv(io.BytesIO(data), **read_csv_kwargs)
    else:
        import pandas as pd
        return pd.read_csv(path, **read_csv_kwargs)
    
def load_parquet(path: str, **read_parquet_kwargs):
    """
    Load a Parquet file into a pandas DataFrame from a local path or a Google Cloud Storage path (gs://...).

    - Local paths are loaded with :pyfunc:`pandas.read_parquet`.
    - GCS paths are downloaded via `google.cloud.storage` and then read into
      pandas using an in-memory buffer.

    Any keyword arguments are forwarded to ``pandas.read_parquet``.

    Raises:
        ValueError: if a GCS path is missing the object name (e.g. only "gs://bucket").
        ImportError: if the google-cloud-storage package is required but not installed.
        RuntimeError: if downloading the GCS object fails.
    """
    if path.startswith("gs://"):
        bucket_name, object_path = _parse_gs_path(path)
        if object_path == "":
            raise ValueError("GCS path must include an object name (gs://bucket/path/to/file.parquet)")

        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)

        try:
            # download as bytes (parquet is binary)
            data = blob.download_as_bytes() if hasattr(blob, "download_as_bytes") else blob.download_as_string()
        except Exception as e:
            raise RuntimeError(f"Failed to download GCS object '{path}': {e}")

        # Lazy import pandas and io to avoid adding heavy deps at module import time
        import io
        import pandas as pd

        return pd.read_parquet(io.BytesIO(data), **read_parquet_kwargs)
    else:
        import pandas as pd
        return pd.read_parquet(path, **read_parquet_kwargs)
