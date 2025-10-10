from abc import ABC, abstractmethod
import pandas as pd
import os
import io
from google.cloud import storage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.exceptions import NotFound

class DatasetProcessor(ABC):
    """
    Abstract base class for formatting diverse skin image datasets into a consistent format.

    @param image_dir: The path to the folder in the `raw` directory in the cloud containing your images. Do not include the bucket name.
    @param bucket_name: The name of the bucket
    @param final_metadata_dir: The folder containing the final metadata file.
    @param final_metadata_file: The name of the file containing the combined metadata from all datasets.
    @param final_image_dir: The path to the folder where the final images are to be stored.
    """
    def __init__(self,
                 bucket_name = "derma-datasets-2", 
                 final_metadata_dir = "final/",
                 final_metadata_file = "metadata_all.csv",
                 final_image_path: str = "final/imgs/"):
        self.bucket_name = bucket_name
        self.final_metadata_dir = final_metadata_dir
        self.final_metadata_file = final_metadata_file
        self.final_image_path = final_image_path
        self.storage_client = storage.Client()

    ######## These are the main functions you will call to transfer your data ########

    def load_metadata(self, metadata_path: str, **kwargs) -> pd.DataFrame:
        """
        Load the metadata file.

        @param metadata_path: Path to the metadata file
        @param **kwargs: Keyword arguments for `pandas.read_csv()` to use when loading the file. Use this, for example, to set the separator.
        """
        print("Loading metadata")
        return self._load_table_from_gcs(metadata_path, **kwargs)

    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Filter the metadata to remove images we do not want to use for our machine learning task.

        For example, this function can be used to filter out images with a low quality score.

        Because this will differ for different datasets, you will need to implement this on your own based on your dataset.

        @param metadata: A Pandas dataframe containing the metadata to filter.
        @returns: A pandas DataFrame with the metadata filtered to only include rows for images you want to do machine learning on.
        """
        pass

    @abstractmethod
    def format_metadata_csv(self, metadata: pd.DataFrame, dataset: str, raw_image_path: str, **kwargs) -> pd.DataFrame:
        """
        Formats a filtered metadata file into a table with the following columns:
            image_id (str): The identifier for the image, as used in your dataset (can be a number, an image hash, etc.)(e.g. 1005, cAINcosoDIOnCIOdID)
            dataset (str): The name of the dataset your metadata corresponds to (e.g. scin, fitzpatrick17k, etc.).
            filename (str): The name of the file containing your image, in the folder `final/imgs` in the Google Cloud Storage bucket.
            orig_filename(str): The name of the original file in the `raw` directory. Used if we need to trace back to original file
            label (str): The label of your image. If your image has multiple labels, pick one to include here, based on the labeling schema you think will be best to use when harmonizing the labels.
            text_desc (str | null): A free text description of the image. Leave as null if your dataset does not contain text descriptions.

        Notes: This function assumes you have already loaded the metadata using `load_metadata` and filtered it using `filter_metadata` if needed.

        @param metadata: A table of metadata.
        @param dataset: The name of your dataset (e.g. fitzpatrick17k, scin, etc.). Use to fill in the `dataset` column.
        @param raw_image_path: The folder in Google Cloud containing the raw images (used to get the original filenames)
        @returns: A DataFrame containing the data to be upserted into metadata.csv.
        """
        pass

    def update_data(self, final_metadata: pd.DataFrame, dataset_meta: str, raw_image_dir: str):
        """
        This function updates the data and metadata in the `final` folder in the bucket. It upserts the `final_metadata` into the metadata file. It also
        copies all the images in the `final_metadata` file into the `final/imgs` folder.

        Notes: 
        * This function in its current implementation assumes the images are named <image_id>.<extension>. If the images are named in a different format,
        please override this function and change the implementation based on your image naming convention.
        * This function also assumes every entry in the `datasets` column of `final_metadata` is the same.

        @param final_metadata: A dataframe containing the final metadata to upsert into the metadata.csv. Must be formatted as described in `format_metadata_csv`.
        @param dataset_meta: The full path of the metadata of the original dataset to copy to the final folder. Ideally, data should be matched to images via some sort of image ID.
        @param raw_image_dir: The directory containing the raw images.
        """
        # update metadata
        print("Updating general metadata")
        self._update_metadata_csv(final_metadata)

        # update images
        print("Copying images")
        image_names = final_metadata['orig_filename'].to_list()
        dataset = final_metadata["dataset"]
        self._update_images(image_names, raw_image_dir, dataset)

        # copy dataset-specific metadata to final folder
        print("Copying dataset specific metadata")
        bucket = self.storage_client.bucket(self.bucket_name)
        source_blob = bucket.blob(dataset_meta)
        destination_blob_name = self.final_metadata_dir.rstrip('/') + '/' + source_blob.name.split('/')[-1]
        bucket.copy_blob(source_blob, bucket, destination_blob_name)

    ######## These are helper funcitons. Theyy are called by the functions above. You can use them to help you override the above functions if needed. ########

    def _update_metadata_csv(self, final_metadata: pd.DataFrame):
        """
        Updates the metadata file with the metadata from your dataset.

        This function considers a unique entry in the metadata.csv file to be a unique combination of image ID and dataset name.
        If your image ID and dataset combination is already in the metadata.csv file, this function updates that entry with the data in final_metadata.
        If your image ID and dataset combination is not in the metadata.csv file, this function adds it to that file.

        If the metadata file does not already exist, it creates it using final_metadata.

        The path to the final metadata file is specified in the constructor of this class.

        @param final_metadata: A dataframe containing the final metadata to upsert into the metadata.csv.
        """
        final_metadata_path = os.path.join(self.final_metadata_dir, self.final_metadata_file)
        try:
            # Try to load existing metadata
            old_meta = self._load_table_from_gcs(final_metadata_path)

            # Set indices to identifiers
            old_meta = old_meta.set_index(['dataset', 'image_id'], drop=False)
            final_metadata = final_metadata.set_index(['dataset', 'image_id'], drop=False)

            # Concatenate and drop duplicates
            upserted = pd.concat([old_meta, final_metadata])
            upserted = upserted[~upserted.index.duplicated(keep='last')]

            # Reset the index
            upserted = upserted.reset_index(drop=True)

        except NotFound:
            # If file doesn't exist, just use the new metadata
            upserted = final_metadata.reset_index(drop=True)

        # Replace old file in Google Cloud with new one (or create new)
        self._write_table_to_gcs(upserted, final_metadata_path)

    def _update_images(self, image_names: list[str], raw_image_dir: str, dataset: str):
        """
        This function copies the images whose filenames are in `image_names` from the `raw_image_dir` to the directory stored in `final_image_path` of this object.

        This function will rename the copied image files in the format <dataset>_<image_id>.<extension>. If an image file already exists in that directory, it will be overriden without warning.

        NOTE: this function assumes images are named as <image_id>.<extension>. If the images are named in a different format, please override this function and update accordingly.

        @param image_names: A list of the filenames of the images you want to copy (with extensions). Image names that don't exist will be skipped.
        @param raw_image_dir: The path to the directory containing the raw images.
        @param dataset: The name of the dataset (e.g. scin, fitzpatrick17k, etc).
        """
        dest_names = [f"{dataset}_{filename}" for filename in image_names]
        self._bulk_copy_files(image_names, raw_image_dir, self.final_image_path, dest_names)

    def _load_table_from_gcs(self, blob_path: str, **kwargs):
        """
        Loads a table file from Google Cloud Storage into a Pandas DataFrame.

        @param blob_path: Path to the file within the bucket.
        @param **kwargs: Keyword arguments forwarded to the pandas read function. E.g. 'sep', 'header', etc.
        @returns: pd.DataFrame: The loaded DataFrame.
        """
        # Get a blob handle
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_path)

        # Download blob content as bytes
        file_data = blob.download_as_bytes()

        # Infer file type from extension for appropriate pandas function
        _, ext = os.path.splitext(blob_path.lower())
        if ext in ['.csv', '.txt']:
            read_func = pd.read_csv
        elif ext in ['.parquet']:
            read_func = pd.read_parquet
        elif ext in ['.xls', '.xlsx']:
            read_func = pd.read_excel
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # For pd.read_csv/read_excel: need BytesIO; for read_parquet, can also use BytesIO
        file_like = io.BytesIO(file_data)
        if read_func is pd.read_csv:
            return read_func(file_like, **kwargs)
        elif read_func is pd.read_parquet:
            return read_func(file_like, **kwargs)
        elif read_func is pd.read_excel:
            return read_func(file_like, **kwargs)
        
    def _write_table_to_gcs(self, df: pd.DataFrame, blob_path: str, **kwargs) -> None:
        """
        Writes a Pandas DataFrame as a CSV file to Google Cloud Storage.

        @param df: The Pandas DataFrame to write.
        @param blob_path: Path to the file within the bucket.
        @param **kwargs: Keyword arguments forwarded to the pandas to_csv function. E.g. 'sep', 'header', etc.
        @returns: None
        """
        # Convert DataFrame to CSV in memory
        csv_buffer = io.BytesIO()
        # Pandas to_csv supports file-like objects; ensure writing bytes (so set encoding)
        df.to_csv(csv_buffer, index=False, **kwargs)
        csv_buffer.seek(0)

        # Get blob
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_path)

        # Upload to GCS
        blob.upload_from_file(csv_buffer, content_type='text/csv')

    def _list_files_in_folder(
        self,
        folder_path: str,
        exclude_dir: bool = False,
        include_prefixes: bool = True
    ) -> list[str]:
        """
        Lists all file names in a folder in Google Cloud Storage.

        @param folder_path: Path to the folder within the bucket (e.g., "my/folder/").
        @param exclude_dir: Whether to exclude directory names from the list.
        @param include_prefixes: If True, includes the prefix folders in the returned file names.
                                If False, returns only the final file names (no folder prefixes).
        @returns: List of file names (full path or base name, according to include_prefixes).
        """
        # Ensure folder_path ends with '/' for correct prefix matching
        if not folder_path.endswith('/'):
            folder_path += '/'

        bucket = self.storage_client.bucket(self.bucket_name)
        # Use the prefix to search
        blobs = self.storage_client.list_blobs(self.bucket_name, prefix=folder_path)
        if exclude_dir:
            files = [blob.name for blob in blobs if not blob.name.endswith('/')]
        else:
            files = [blob.name for blob in blobs]

        if not include_prefixes:
            # Only return the base file names (strip folder_path prefix)
            # Protects against empty folder_path or accidental mismatches
            cut_len = len(folder_path)
            files = [f[cut_len:] if f.startswith(folder_path) else f for f in files]

        return files

    def _bulk_copy_files(self, file_list: list[str], src_dir: str, dst_dir: str, dest_names: list[str]=None, max_workers: int=20):
        """
        Bulk copies files from one directory to another within a Google Cloud Storage bucket,
        optionally renaming them, and prints a progress bar.

        @param file_list: List of file paths (relative to src_dir) to copy.
        @param src_dir: Source directory path (e.g., 'folder1/subfolder1/').
        @param dst_dir: Destination directory path (e.g., 'folder2/subfolder2/').
        @param dest_names: (Optional) List of destination filenames (relative to dst_dir).
                           If given, must be same length as file_list.
        @param max_workers: Number of worker threads (default 20).
        @returns: None
        """
        if dest_names is not None and len(dest_names) != len(file_list):
            raise ValueError("dest_names must be the same length as file_list.")

        bucket = self.storage_client.bucket(self.bucket_name)
        if not src_dir.endswith('/'): src_dir += '/'
        if not dst_dir.endswith('/'): dst_dir += '/'

        def do_copy(src_fname, dst_fname):
            src_blob_path = src_dir + src_fname
            dst_blob_path = dst_dir + dst_fname
            source_blob = bucket.blob(src_blob_path)
            bucket.copy_blob(source_blob, bucket, dst_blob_path)
            return dst_blob_path

        # Prepare pairs of (src, dest) filenames
        if dest_names is not None:
            copy_pairs = zip(file_list, dest_names)
        else:
            copy_pairs = ((fname, fname) for fname in file_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(do_copy, src, dst) for src, dst in copy_pairs]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying files", unit="file"):
                pass  # Just update the progress bar



    




