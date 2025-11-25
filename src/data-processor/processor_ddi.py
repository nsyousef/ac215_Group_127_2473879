from processor import DatasetProcessor
import pandas as pd
import os


class DatasetProcessorDDI(DatasetProcessor):
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
        print("Formatting metadata")
        img_names, img_ids = self._get_img_ids_names(raw_image_path)

        id_name_map = pd.DataFrame({"orig_filename": img_names}, index=img_ids)
        print(id_name_map.head())

        final_metadata_drft = pd.DataFrame()
        final_metadata_drft["image_id"] = id_name_map.index
        final_metadata_drft["dataset"] = [dataset] * final_metadata_drft.shape[0]
        final_metadata_drft["orig_filename"] = metadata["DDI_file"]
        final_metadata_drft["filename"] = final_metadata_drft.apply(
            lambda x: f"{x['dataset']}_{x['image_id']}{os.path.splitext(x['orig_filename'])[1]}", axis=1
        )
        final_metadata_drft["label"] = metadata["disease"]

        # Add malignant metadata as text description
        final_metadata_drft["text_desc"] = [None] * final_metadata_drft.shape[0]
        malignant_list = metadata["malignant"].tolist()
        assert len(malignant_list) == len(final_metadata_drft)

        for i, malig_bool in enumerate(malignant_list):
            if malig_bool:
                malig_sentence = "This lesion is malignant."
            else:
                malig_sentence = "This lesion is benign."
            final_metadata_drft.loc[final_metadata_drft.index[i], "text_desc"] = malig_sentence

        # order columns
        final_metadata = final_metadata_drft[["image_id", "dataset", "filename", "orig_filename", "label", "text_desc"]]

        return final_metadata

    def _list_files_in_folder(
        self, folder_path: str, exclude_dir: bool = False, include_prefixes: bool = True
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
        if not folder_path.endswith("/"):
            folder_path += "/"

        bucket = self.storage_client.bucket(self.bucket_name)
        # Use the prefix to search
        blobs = self.storage_client.list_blobs(self.bucket_name, prefix=folder_path)
        if exclude_dir:
            files = [blob.name for blob in blobs if not blob.name.endswith("/") and not blob.name.endswith(".csv")]
        else:
            files = [blob.name for blob in blobs if not blob.name.endswith(".csv")]

        if not include_prefixes:
            # Only return the base file names (strip folder_path prefix)
            # Protects against empty folder_path or accidental mismatches
            cut_len = len(folder_path)
            files = [f[cut_len:] if f.startswith(folder_path) else f for f in files]

        return files


if __name__ == "__main__":
    dp = DatasetProcessorDDI()
    DDI_BASE_PATH = "raw/ddidiversedermatologyimages/"
    DDI_META_PATH = os.path.join(DDI_BASE_PATH, "ddi_metadata.csv")
    DDI_IMG_PATH = DDI_BASE_PATH
    metadata = dp.load_metadata(DDI_META_PATH)
    metadata_ins = dp.format_metadata_csv(metadata, "ddi", DDI_IMG_PATH)
    dp.update_data(metadata_ins, metadata, "ddi_metadata.csv", DDI_IMG_PATH)
