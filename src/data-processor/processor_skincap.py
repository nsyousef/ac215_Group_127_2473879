from processor import DatasetProcessor
import pandas as pd
import os
from google.api_core.exceptions import NotFound

class DatasetProcessorSkinCAP(DatasetProcessor):
    def filter_metadata(self, metadata: pd.DataFrame, raw_image_path: str) -> pd.DataFrame:
        """
        Filter the metadata to remove images where the quality is scored as "wrongly labelled".

        Also filter metadata to exclude any images for which we don't actually have the files.

        @param metadata: A Pandas dataframe containing the metadata to filter.
        @param raw_image_path: The path to the folder containing the raw images.
        @returns: A pandas DataFrame with the metadata filtered to only include rows for images you want to do machine learning on.
        """
        pass
    
    def format_metadata_csv(self, metadata: pd.DataFrame, dataset: str, raw_image_path: str) -> pd.DataFrame:
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
        @param raw_image_path: The path to the directory containing raw images.
        @returns: A DataFrame containing the data to be upserted into metadata_all.csv.
        """
        pass

    def update_data(self, metadata: pd.DataFrame, metadata_name: str):
        """
        This function updates the metadata in the `final` folder in the bucket by adding captions to corresponding Fitzpatrick17k and DDI images 
        and it writes the SkinCAP metadata to the `final` folder.

        @param metadata: A dataframe containing the final metadata to add metadata_all.csv.
        @param metadata_name: The name to call the SkinCAP metadata.
        """
        # update metadata
        print("Updating general metadata")
        self._update_metadata_csv(metadata)

        # copy dataset-specific metadata to final folder
        print("Copying dataset specific metadata")
        self._write_table_to_gcs(metadata, os.path.join(self.final_metadata_dir, metadata_name))

    def _update_metadata_csv(self, metadata: pd.DataFrame):
        """
        Updates the metadata_all file with the caption from your SkinCAP.

        @param metadata: A dataframe containing the final metadata to add metadata_all.csv.
        """
        final_metadata_path = os.path.join(self.final_metadata_dir, self.final_metadata_file)
        try:
            # Try to load existing metadata
            old_meta = self._load_table_from_gcs(final_metadata_path)

            metadata = metadata.rename(columns={"ori_file_path": "orig_filename"})
            metadata = metadata[["orig_filename", "caption_zh_polish_en"]]
            new_meta = pd.merge(old_meta, metadata, on='orig_filename', how='left')
            # Add new caption to existing text_desc
            new_meta['text_desc'] = new_meta['text_desc'].fillna('') + ' ' + new_meta['caption_zh_polish_en'].fillna('')
            # Clean up any leading/trailing whitespace
            new_meta['text_desc'] = new_meta['text_desc'].str.strip()
            new_meta = new_meta.drop(columns=['caption_zh_polish_en'])
        except NotFound as e:
            raise FileNotFoundError(f"File {final_metadata_path} not found in GCS") from e
        
        # Replace old file in Google Cloud with new one (or create new)
        self._write_table_to_gcs(new_meta, final_metadata_path)

if __name__ == "__main__":
    dp = DatasetProcessorSkinCAP()
    SKINCAP_BASE_PATH = "raw/SkinCAP/"
    SKINCAP_META_PATH = os.path.join(SKINCAP_BASE_PATH, "skincap_v240623.csv")
    SKINCAP_IMG_PATH = os.path.join(SKINCAP_BASE_PATH, "skincap")
    metadata = dp.load_metadata(SKINCAP_META_PATH)
    dp.update_data(metadata, "skincap_metadata.csv")
