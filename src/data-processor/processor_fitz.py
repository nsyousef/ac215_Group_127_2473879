from processor import DatasetProcessor
import pandas as pd
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

class DatasetProcessorFitz(DatasetProcessor):
    def expand_raw_fitz_from_skincap(self, scap_meta: pd.DataFrame, fitz_raw_imgs_dir: str, scap_raw_imgs_dirs: list[str]):
        """
        This function expands the raw Fitzpatrick data with the images in the SkinCAP dataset.

        The SkinCAP dataset includes images from the Fitzpatrick dataset that are not able to be accessed via URL.
        This function copies those images from the SkinCAP folder to the Fitzpatrick raw folder and renames them appropriately.
        The reason I do it this way is so the rest of the pipeline can just worry about the raw Fitzpatrick images, without having to straddle two datasets.
        In this way, if we ever remove the SkinCAP data, we won't need to rewrite this whole pipeline. We will just need to remove this step in the future and it will work.

        @param fitz_meta: The raw metadata file for the Fitzpatrick dataset, as a Pandas DataFrame
        @param scap_meta: The raw metadata file for the SkinCAP dataset, as a Pandas DataFrame
        @param fitz_raw_imgs_dir: The directory where the raw Fitzpatrick images are stored
        @param scap_raw_imgs_dirs: A list of directories where the raw SkinCAP images are stored
        @returns: None. This function does not return anything. It just copies the images into the raw Fitzpatrick folder.
        """

        # record the names of the images in each of the two SkinCAP folders mapped to their paths
        all_scap_paths = []
        all_scap_ids = []
        for scap_dir in scap_raw_imgs_dirs:
            fitz_filepaths, fitz_img_ids = self._get_img_ids_names(scap_dir, include_prefixes=True)
            # drop .DS_Store
            fitz_filepaths = [path for path in fitz_filepaths if not path.endswith(".DS_Store")]
            fitz_img_ids = [id for id in fitz_img_ids if not id.endswith(".DS_Store")]
            all_scap_paths += fitz_filepaths
            all_scap_ids += fitz_img_ids

        scap_name_path_map = pd.DataFrame({"scap_img_id": all_scap_ids, "scap_img_path": all_scap_paths})

        print(f"{scap_name_path_map.shape[0]} SkinCAP images found.")

        # if two images have same ID, only keep one copy
        scap_name_path_map = scap_name_path_map.drop_duplicates(subset=["scap_img_id"])
        print(f"{scap_name_path_map.shape[0]} SkinCAP images left after deduplicating by ID.")
        print(scap_name_path_map.head())

        # only keep necessary columns of SkinCAP metadata
        scap_meta = scap_meta[["id", "skincap_file_path", "ori_file_path", "source"]]

        # add md5hash to skincap metadata
        scap_meta["md5hash"] = scap_meta["ori_file_path"].apply(lambda x: os.path.splitext(x)[0])

        # merge filepaths with SkinCAP metadata
        scap_meta = scap_meta.merge(scap_name_path_map, left_on="id", right_on="scap_img_id", validate="1:1").drop("id", axis=1)

        # filter to only include SkinCAP images from Fitzpatrick
        scap_in_fitz_flg = scap_meta["source"] == "fitzpatrick17k"
        scap_meta = scap_meta[scap_in_fitz_flg]
        print(f"{scap_meta.shape[0]} SkinCAP images are from the Fitzpatrick dataset.")

        # get list of Fitzpatrick images we have
        fitz_filepaths, fitz_img_ids = self._get_img_ids_names(fitz_raw_imgs_dir, include_prefixes=True)
        fitz_name_path_map = pd.DataFrame({"fitz_image_id": fitz_img_ids, "fitz_image_filepath": fitz_filepaths})
        print(f"We have {fitz_name_path_map.shape[0]} images in our Fitzpatrick raw data folder before adding SkinCAP images.")

        # merge Fitzpatrick images with SkinCAP mapping
        full_mapping = scap_meta.merge(fitz_name_path_map, how="outer", on="md5hash", suffixes=["scap_", "fitz_"])

        # filter for images for which we don't already have Fitzpatrick raw data (we only need to copy these)
        no_fitz_raw_flg = full_mapping["fitz_image_id"].isna()
        full_mapping = full_mapping[no_fitz_raw_flg]
        print(full_mapping.head())

        # strip full_mapping down to a simple map from source to destination

        # get extension of SkinCAP image
        full_mapping["extension"] = full_mapping["scap_img_path"].apply(lambda x: os.path.splitext(x)[1])

        # construct final path for image to go
        full_mapping["dest_dir"] = [fitz_raw_imgs_dir] * full_mapping.shape[0]
        full_mapping["dest_path"] = full_mapping.apply(lambda x: os.path.join(x['dest_dir'], f"{x['md5hash']}{x['extension']}"), axis=1)

        # simplify full_mapping
        full_mapping = full_mapping[['scap_img_path', 'dest_path']]

        # sanity checks (remove if it turns out SkinCAP dataset does not have all PNGs)
        # NOTE: the purpose of these is to abort the copy if there is a bug in this code
        assert full_mapping['scap_img_path'].str.endswith(".png").all(), "There are non-PNG images in the SkinCAP dataset"
        assert full_mapping['dest_path'].str.endswith(".png").all(), "File extensions were changed incorrectly"

        full_mapping.to_csv("tmp_full_mapping.csv")

        # perform the copy
        # self._bulk_copy_blobs(full_mapping["scap_img_path"].to_list(), full_mapping["dest_path"].to_list())

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
        final_metadata["filename"] = final_metadata.apply(lambda x: f"{x['dataset']}_{x['image_id']}{os.path.splitext(x['orig_filename'])[1]}", axis=1)
        # NOTE: I am choosing this category since our goal is to have an ML app that identifies the disease and gives advice on it.
        # I think the collapsed categories are too broad for the model to be able to identify the disease and give good advice about it.
        final_metadata["label"] = metadata["label"]
        final_metadata["text_desc"] = [None] * final_metadata.shape[0]
        
        print(final_metadata.head())

        # order columns
        final_metadata = final_metadata[["image_id", "dataset", "filename", "orig_filename", "label", "text_desc"]]

        return final_metadata
    
    def _compute_md5_for_blob(self, blob):
        md5 = hashlib.md5()
        with blob.open("rb") as f:
            while True:
                data = f.read(8192)
                if not data:
                    break
                md5.update(data)
        return blob.name, md5.hexdigest()

    def _compute_md5_gcs_list_parallel(self, bucket_name: str, file_list: list[str], max_workers: int=16):
        """
        Compute the MD5 hashes of a list of files (i.e. Google Cloud Storage blobs) in parallel using file streaming to save RAM.

        @param bucket_name: The name of the bucket in which the files are stored.
        @param file_list: A list of full file paths (from the root of the bucket) to all the files you want to hash.
        @param max_workers: The maximum number of parallel workers to use.
        """
        client = self.storage_client
        bucket = client.bucket(bucket_name)
        # Prepare Blob objects for the paths provided
        blobs = [bucket.blob(file_name) for file_name in file_list]
        hashes = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_blob = {executor.submit(self.compute_md5_for_blob, blob): blob for blob in blobs}
            for future in as_completed(future_to_blob):
                name, md5sum = future.result()
                hashes[name] = md5sum

        return hashes

if __name__ == "__main__":
    dp = DatasetProcessorFitz()
    FITZ_BASE_PATH = "raw/fitzpatrick17k/"
    SKINCAP_BASE_PATH = "raw/SkinCAP"
    FITZ_META_PATH = os.path.join(FITZ_BASE_PATH, "fitzpatrick17k.csv")
    SKINCAP_META_PATH = os.path.join(SKINCAP_BASE_PATH, "skincap_v240623.csv")
    FITZ_IMG_PATH = os.path.join(FITZ_BASE_PATH, "images")
    SKINCAP_IMG_PATH_1 = os.path.join(SKINCAP_BASE_PATH, "skincap")
    SKINCAP_IMG_PATH_2 = os.path.join(SKINCAP_BASE_PATH, "skincap/not_include")
    fitz_metadata = dp.load_metadata(FITZ_META_PATH)
    scap_metadata = dp.load_metadata(SKINCAP_META_PATH)
    # This function copies extra images from the SkinCAP directories into the RAW data directory for Fitzpatrick.
    dp.expand_raw_fitz_from_skincap(scap_metadata, FITZ_IMG_PATH, [SKINCAP_IMG_PATH_1, SKINCAP_IMG_PATH_2])
    # Filter and copy everything to final folder
    metadata_filt = dp.filter_metadata(fitz_metadata, FITZ_IMG_PATH)
    metadata_ins = dp.format_metadata_csv(metadata_filt, "fitzpatrick17k", FITZ_IMG_PATH)
    dp.update_data(metadata_ins, metadata_filt, "fitzpatrick17k_metadata.csv", FITZ_IMG_PATH)
