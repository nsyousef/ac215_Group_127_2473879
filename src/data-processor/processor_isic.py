from processor import DatasetProcessor
import pandas as pd
import os

class DatasetProcessorIsic(DatasetProcessor):
    def filter_metadata(self, metadata: pd.DataFrame):
        """
        Filter the ISIC metadata to only include: 
        * rows where diagnosis_3 is not null
        * rows where the image type is in ('clinical: close-up', 'TBP tile: close-up', 'clinical: overview')

        @param metadata: A DataFrame of the original metadata.
        @returns: The filtered metadata.
        """
        # drop columns not needed
        columns_to_drop = ['attribution', "copyright_license", "acquisition_day"]
        metadata_filt = metadata.drop(columns=columns_to_drop)

        # filter rows
        valid_types = [
            "clinical: close-up",
            "TBP tile: close-up",
            "clinical: overview",
        ]

        keep_flgs = metadata_filt["diagnosis_3"].notna() & metadata_filt['image_type'].astype(str).str.strip().isin(valid_types)
        metadata_filt = metadata_filt[keep_flgs]

        print("Filtered metadata shape:")
        print(metadata_filt.shape)

        return metadata_filt

    def format_metadata_csv(self, metadata, dataset, raw_image_path, **kwargs):
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
        # get data needed to add original filenames
        raw_img_filenames, img_ids = self._get_img_ids_names(raw_image_path)
        img_id_name_map = pd.DataFrame({"orig_filename": raw_img_filenames}, index=img_ids)

        # construct formatted metadata
        form_met = pd.DataFrame()
        form_met["image_id"] = metadata['isic_id']
        form_met["dataset"] = [dataset] * form_met.shape[0]
        form_met["label"] = metadata['diagnosis_3']

        # Include metadata in text descriptions
        form_met["text_desc"] = [None] * form_met.shape[0]
        sex_list = metadata['sex'].tolist()
        age_list = metadata['age_approx'].tolist()
        anatom_general_list = metadata['anatom_site_general'].tolist()
        anatom_special_list = metadata['anatom_site_special'].tolist()

        for i in range(len(metadata)):
            new_caption = ''
            
            if str(sex_list[i]) != 'nan':
                gender_sentence = f' The sex is {sex_list[i]}.'
                new_caption += gender_sentence
            
            if str(age_list[i]) != 'nan':
                age_sentence = f' The approximate age is {int(age_list[i])}.'
                new_caption += age_sentence
            
            if str(anatom_general_list[i]) != 'nan':
                location_sentence = f' The body location is {anatom_general_list[i]}.'
                new_caption += location_sentence
            elif str(anatom_special_list[i]) != 'nan':
                location_sentence = f' The body location is {anatom_special_list[i]}.'
                new_caption += location_sentence
            
            form_met.loc[form_met.index[i], 'text_desc'] = new_caption.strip()

        # add original and final filenames
        form_met = form_met.merge(img_id_name_map, how='left', left_on='image_id', right_index=True)
        form_met["extension"] = form_met['orig_filename'].apply(lambda x: os.path.splitext(x)[1])
        form_met["filename"] = form_met.apply(lambda x: f"{x['dataset']}_{x['image_id']}{x['extension']}", axis=1)

        # order columns
        form_met = form_met[["image_id", "dataset", "filename", "orig_filename", "label", "text_desc"]]

        return form_met

if __name__ == "__main__":
    dp = DatasetProcessorIsic()
    BASE_PATH = "raw/isic/isic/"
    META_PATH = os.path.join(BASE_PATH, "metadata.csv")
    IMG_PATH = BASE_PATH
    metadata = dp.load_metadata(META_PATH)
    metadata_filt = dp.filter_metadata(metadata)
    metadata_ins = dp.format_metadata_csv(metadata_filt, "isic", IMG_PATH)
    dp.update_data(metadata_ins, metadata_filt, "isic_metadata.csv", IMG_PATH)
