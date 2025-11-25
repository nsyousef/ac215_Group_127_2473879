"""
Verify pipeline output artifacts
"""

import os
from google.cloud import storage


def verify_pipeline_output():
    print("=== PIPELINE OUTPUT VERIFICATION ===")

    # Check if metadata file exists in GCS
    try:
        client = storage.Client()
        bucket = client.bucket("derma-datasets-2")
        blob = bucket.blob("final/metadata_all_harmonized.csv")

        if blob.exists():
            blob.reload()
            print("Harmonized metadata file found in GCS")
            print(f"    Size: {blob.size} bytes")
            print(f"    Updated: {blob.time_created}")
        else:
            print("Metadata file not found in GCS")
    except Exception as e:
        print(f"Could not verify GCS output: {e}")


if __name__ == "__main__":
    verify_pipeline_output()
