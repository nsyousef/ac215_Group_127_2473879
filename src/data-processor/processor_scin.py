from processor import DatasetProcessor
import pandas as pd
import os
from google.api_core.exceptions import NotFound
import ast


def filter_highest_confidence_labels(row):
    """
    Keep only labels with the highest confidence score.
    Returns tuple of (comma-separated string of labels, max confidence as int).
    """
    labels = row["dermatologist_skin_condition_on_label_name"]
    confidences = row["dermatologist_skin_condition_confidence"]

    # Parse if they're strings
    if isinstance(labels, str):
        labels = ast.literal_eval(labels)
    if isinstance(confidences, str):
        confidences = ast.literal_eval(confidences)

    # Handle empty cases
    if not labels or not confidences:
        return "", 0

    # Get the maximum confidence
    max_confidence = max(confidences)

    # Keep only labels with max confidence
    highest_labels = [label for label, conf in zip(labels, confidences) if conf == max_confidence]

    # Return as tuple: (labels string, max confidence as int)
    return ", ".join(highest_labels), int(max_confidence)


class DatasetProcessorScin(DatasetProcessor):
    def filter_metadata(self, metadata: pd.DataFrame, text_metadata: pd.DataFrame, raw_image_path: str) -> pd.DataFrame:
        """
        Filter the metadata to remove images where the quality is scored as "wrongly labelled".

        Also filter metadata to exclude any images for which we don't actually have the files.

        @param metadata: A Pandas dataframe containing the metadata to filter.
        @param raw_image_path: The path to the folder containing the raw images.
        @returns: A pandas DataFrame with the metadata filtered to only include rows for images you want to do machine learning on.
        """
        metadata = metadata[~(metadata["dermatologist_skin_condition_on_label_name"] == "[]")]
        metadata.set_index("case_id", inplace=True)
        metadata[["labels", "max_confidence"]] = metadata.apply(
            filter_highest_confidence_labels, axis=1, result_type="expand"
        )

        text_metadata.set_index("case_id", inplace=True)
        text_metadata = text_metadata.loc[metadata.index]

        combined_df = metadata.merge(text_metadata, left_index=True, right_index=True)

        rows = []

        for idx, row in combined_df.iterrows():
            # Process each image path column
            for img_col in ["image_1_path", "image_2_path", "image_3_path"]:
                if img_col in row and pd.notna(row[img_col]) and row[img_col] != "":
                    # Extract image_id: part between 'images/' and '.png'
                    path = row[img_col]
                    if "images/" in path:
                        image_id = path.split("images/")[-1].replace(".png", "")

                        if image_id == "-2243186711511406658":  # Image for this is missing
                            continue

                        # Copy all columns from original row
                        new_row = row.to_dict()

                        # Add/update image-specific columns
                        new_row["image_id"] = image_id
                        new_row["dataset"] = "SCIN"
                        new_row["label"] = row["labels"]
                        new_row["filename"] = "SCIN" + "_" + image_id + ".png"
                        new_row["orig_filename"] = image_id + ".png"
                        new_row["confidence"] = row["max_confidence"]

                        rows.append(new_row)

        # Create new dataframe
        combined_image_df_full = pd.DataFrame(rows)
        combined_image_df_full.set_index("image_id", inplace=True)
        old_len = len(combined_image_df_full)
        combined_image_df_full = combined_image_df_full[combined_image_df_full["confidence"] > 2]
        new_len = len(combined_image_df_full)
        print(f"Filtered out {old_len - new_len} images due to low confidence")

        combined_image_df = combined_image_df_full[["dataset", "label", "filename", "orig_filename", "confidence"]]

        return combined_image_df_full, combined_image_df

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
        return metadata

    def update_data(self, metadata: pd.DataFrame, text_metadata: pd.DataFrame, metadata_name: str):
        """
        This function updates the metadata in the `final` folder in the bucket by adding captions to corresponding Fitzpatrick17k and DDI images
        and it writes the SkinCAP metadata to the `final` folder.

        @param metadata: A dataframe containing the final metadata to add metadata_all.csv.
        @param text_metadata: A dataframe containing the text metadata to add metadata_all.csv.
        @param metadata_name: The name to call the SkinCAP metadata.
        """
        # update metadata
        print("Updating general metadata")
        self._update_metadata_csv(metadata)

        # copy dataset-specific metadata to final folder
        print("Copying dataset specific metadata")
        self._write_table_to_gcs(text_metadata, os.path.join(self.final_metadata_dir, metadata_name))

    def _update_metadata_csv(self, metadata: pd.DataFrame):
        """
        Updates the metadata_all file with the caption from your SkinCAP.

        @param metadata: A dataframe containing the final metadata to add metadata_all.csv.
        """
        final_metadata_path = os.path.join(self.final_metadata_dir, self.final_metadata_file)
        new_meta = metadata
        # Replace old file in Google Cloud with new one (or create new)
        self._write_table_to_gcs(new_meta, final_metadata_path)


if __name__ == "__main__":
    SKIN_BASE_PATH = "raw/scin/dx-scin-public-data/dataset"
    SKIN_META_PATH = os.path.join(SKIN_BASE_PATH, "scin_labels.csv")
    SCIN_TEXT_META_PATH = os.path.join(SKIN_BASE_PATH, "scin_cases.csv")
    SCIN_IMG_PATH = os.path.join(SKIN_BASE_PATH, "images")
    data_processor = DatasetProcessorScin()
    labels_metadata = data_processor.load_metadata(SKIN_META_PATH)
    text_metadata = data_processor.load_metadata(SCIN_TEXT_META_PATH)
    df_full, df = data_processor.filter_metadata(labels_metadata, text_metadata, SCIN_IMG_PATH)
    data_processor.update_data(df, "scin_metadata.csv")
