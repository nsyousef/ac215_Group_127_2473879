from processor import DatasetProcessor
import pandas as pd
import os

class DatasetProcessorFitz(DatasetProcessor):
    def filter_metadata(self, metadata: pd.DataFrame, raw_image_path: str) -> pd.DataFrame:
        """
        Filter the metadata to remove images where the quality is scored as "wrongly labelled".

        Also filter metadata to exclude any images for which we don't actually have the files.

        @param metadata: A Pandas dataframe containing the metadata to filter.
        @param raw_image_path: The path to the folder containing the raw images.
        @returns: A pandas DataFrame with the metadata filtered to only include rows for images you want to do machine learning on.
        """
        print("Filtering metadata")
        # get the list of images we have
        _, img_ids = self._get_img_ids_names(raw_image_path)

        print("Image IDs:")
        print(img_ids[0:10])

        # filter data
        print(f"Original shape: {metadata.shape}")
        keep_flg = (metadata["qc"] != "3 Wrongly labelled") & metadata["md5hash"].isin(img_ids)
        metadata_filt = metadata[keep_flg]
        print(f"Final shape: {metadata_filt.shape}")
        return metadata_filt
    
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

        final_metadata = pd.DataFrame()
        final_metadata["image_id"] = metadata["md5hash"]
        # add original filename column
        final_metadata = final_metadata.merge(id_name_map, how="left", left_on="image_id", right_index=True)
        final_metadata["dataset"] = [dataset] * final_metadata.shape[0]
        final_metadata["filename"] = final_metadata.apply(lambda x: f"{x['dataset']}_{x['image_id']}{os.path.splitext(x["orig_filename"])[1]}", axis=1)
        # NOTE: I am choosing this category since our goal is to have an ML app that identifies the disease and gives advice on it.
        # I think the collapsed categories are too broad for the model to be able to identify the disease and give good advice about it.
        final_metadata["label"] = metadata["label"]
        final_metadata["text_desc"] = [None] * final_metadata.shape[0]
        
        print(final_metadata.head())

        # order columns
        final_metadata = final_metadata[["image_id", "dataset", "filename", "orig_filename", "label", "text_desc"]]

        return final_metadata

if __name__ == "__main__":
    dp = DatasetProcessorFitz()
    FITZ_BASE_PATH = "raw/fitzpatrick17k/"
    FITZ_META_PATH = os.path.join(FITZ_BASE_PATH, "fitzpatrick17k.csv")
    FITZ_IMG_PATH = os.path.join(FITZ_BASE_PATH, "images")
    metadata = dp.load_metadata(FITZ_META_PATH)
    metadata_filt = dp.filter_metadata(metadata, FITZ_IMG_PATH)
    metadata_ins = dp.format_metadata_csv(metadata_filt, "fitzpatrick17k", FITZ_IMG_PATH)
    dp.update_data(metadata_ins, metadata_filt, "fitzpatrick17k_metadata.csv", FITZ_IMG_PATH)


