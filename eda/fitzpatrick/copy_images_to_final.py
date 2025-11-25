from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==== Configure these ====
BUCKET_NAME = 'derma-datasets-2'
SRC_PREFIX = 'raw/fitzpatrick17k/images/'
DEST_PREFIX = 'final/imgs/'
FILENAME_PREFIX = 'fitzpatrick17k_'
MAX_WORKERS = 8   # Adjust as needed

def copy_blob_with_prefix(blob_name, storage_client):
    bucket = storage_client.bucket(BUCKET_NAME)
    src_blob = bucket.blob(blob_name)
    # Get only the filename
    filename = blob_name.rsplit('/', 1)[-1]
    dest_blob_name = f"{DEST_PREFIX}{FILENAME_PREFIX}{filename}"
    # Perform server-side copy (fast)
    bucket.copy_blob(src_blob, bucket, dest_blob_name)
    return blob_name, dest_blob_name

def main():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    print("Listing source files...")
    blobs = bucket.list_blobs(prefix=SRC_PREFIX)
    blob_names = [blob.name for blob in blobs if not blob.name.endswith('/')]  # skip folders

    n = len(blob_names)
    print(f"Found {n} files. Copying in parallel...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(copy_blob_with_prefix, blob_name, storage_client) for blob_name in blob_names]
        with tqdm(total=n, desc="Copying", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    src, dst = future.result()
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    pbar.update(1)

if __name__ == "__main__":
    main()
