from processor import DatasetProcessor
import pandas as pd
import os
import json
from google.cloud import storage

class DatasetProcessorDerm1M(DatasetProcessor):
    def filter_metadata(self, metadata: pd.DataFrame):
        """
        Filter the Derm1M metadata to only include: 
        * rows where the labels match terms in ontology.json
        * rows where the image type is in ('clinical: close-up', 'TBP tile: close-up', 'clinical: overview')

        @param metadata: A DataFrame of the original metadata.
        @returns: The filtered metadata.
        """
        client = storage.Client()
        bucket = client.bucket('derma-datasets-2')
        blob = bucket.blob('raw/Derm1M/ontology.json')
        json_str = blob.download_as_text()
        ontology = json.loads(json_str)

        ontology_terms = []
        for key, values in ontology.items():
            ontology_terms.append(key.lower())
            for value in values:
                ontology_terms.append(value.lower())
        ontology_terms_set = set(ontology_terms)

        df_filtered = metadata[~metadata.select_dtypes(include=['object']).apply(
            lambda x: x.astype(str).str.contains(r'\bdermosc\w*', case=False, na=False, regex=True)
        ).any(axis=1)]
        labels = (
            df_filtered['hierarchical_disease_label']
            .dropna()
            .apply(lambda x: x.split(',')[-1].strip().lower() if isinstance(x, str) else '')
        )
        mask = labels.isin(ontology_terms_set)
        df_filtered = metadata.loc[labels.index[mask]]

        return df_filtered

    def format_metadata_csv(self, metadata, dataset, **kwargs):
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
        # construct formatted metadata
        form_met = pd.DataFrame()
        form_met['image_id'] = metadata['filename'].apply(lambda x: os.path.splitext(x)[0]).str.replace('/', '_')
        form_met["dataset"] = [dataset] * form_met.shape[0]
        form_met["orig_filename"] = metadata['filename']
        form_met["filename"] = form_met.apply(
            lambda x: f"{x['dataset']}_{x['image_id']}{os.path.splitext(x['orig_filename'])[1]}", axis=1
        )

        # Add labels and ontology data
        form_met['label'] = metadata['hierarchical_disease_label'].apply(lambda x: x.split(',')[-1])

        split_cols = metadata['hierarchical_disease_label'].str.split(',', expand=True)
        split_cols.columns = [f'level_{i+1}' for i in range(len(split_cols.columns))]
        form_met = pd.concat([form_met, split_cols], axis=1)

        # Include metadata in text descriptions
        gender_list = metadata['gender'].tolist()
        age_list = metadata['age'].tolist()
        location_list = metadata['body_location'].tolist()
        symptoms_list = metadata['symptoms'].tolist()
        assert len(gender_list) == len(metadata)
        assert len(age_list) == len(metadata)
        assert len(location_list) == len(metadata)
        assert len(symptoms_list) == len(metadata)

        for i in range(len(metadata)):
            new_caption = str(metadata.iloc[i]['caption'])

            if gender_list[i] != 'No gender information' and gender_list[i] != 'male, female':
                gender_sentence = f' The gender is {gender_list[i]}.'
                new_caption += gender_sentence

            if age_list[i] != 'No age information':
                age_sentence = f' The age is {age_list[i]}.'
                new_caption += age_sentence

            if location_list[i] != 'No body location information':
                location_sentence = f' The body location is {location_list[i]}.'
                new_caption += location_sentence

            if symptoms_list[i] != 'No symptom information':
                symptom_sentence = f' The symptoms are {symptoms_list[i]}.'
                new_caption += symptom_sentence

            metadata.loc[metadata.index[i], 'caption'] = new_caption

        form_met["text_desc"] = metadata['caption']

        # order columns
        level_cols = [col for col in form_met.columns if col.startswith('level_')]
        base_cols = ["image_id", "dataset", "filename", "orig_filename", "label", "text_desc"]
        form_met = form_met[base_cols + level_cols]

        return form_met
    
    def _update_images(self, image_names: list[str], raw_image_dir: str, dataset: str):
        """
        This function copies the images whose filenames are in `image_names` from the `raw_image_dir` to the directory stored in `final_image_path` of this object.

        This function will rename the copied image files in the format <dataset>_<image_id>.<extension>. If an image file already exists in that directory, it will be overriden without warning.

        NOTE: this function assumes images are named as <image_id>.<extension>. If the images are named in a different format, please override this function and update accordingly.

        @param image_names: A list of the filenames of the images you want to copy (with extensions). Image names that don't exist will be skipped.
        @param raw_image_dir: The path to the directory containing the raw images.
        @param dataset: The name of the dataset (e.g. scin, fitzpatrick17k, etc).
        """
        clean_image_names = [name.replace('/', '_') for name in image_names]
        dest_names = [f"{dataset}_{filename}" for filename in clean_image_names]
        self._bulk_copy_files(image_names, raw_image_dir, self.final_image_path, dest_names)


if __name__ == "__main__":
    dp = DatasetProcessorDerm1M()
    BASE_PATH = "raw/Derm1M"
    META_PATH = os.path.join(BASE_PATH, "Derm1M_v2_pretrain.csv")
    IMG_PATH = BASE_PATH
    metadata = dp.load_metadata(META_PATH)
    metadata_filt = dp.filter_metadata(metadata)
    metadata_ins = dp.format_metadata_csv(metadata_filt, "derm1m")
    dp.update_data(metadata_ins, metadata_filt, "derm1m_metadata.csv", IMG_PATH)
