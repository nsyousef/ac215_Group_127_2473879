# data-processor

The purpose of this microservice is to convert the raw datasets into a standardized format and conduct label harmonization. The final output file, `metadata_all_harmonized.csv`,  contains all the formatted metadata that is needed to start model training.

## Instructions to Run the data-processor Pipeline

1) `cd` into the `src/data-processor/` folder.
2) Run `./data-processor.sh`

This script will automatically run all dataset processors in sequence (DDI, Fitzpatrick, SkinCap, ISIC, Derm1M), then harmonize labels and filter descriptions, and finally verify the output. The complete pipeline processes all datasets and produces `metadata_all_harmonized.csv` in the `final/` directory of the GCS bucket.

## Instructions to Create a Processor Script

1) Please create a file called `processor_<your dataset name or abbrev>.py` with a class that is a child of `DatasetProcessor` (in `processor.py).
2) Override and implement at least the `filter_metadata` and `format_metadata_csv` functions. You may also override other functions (including helper functions) as needed to make the process work for your dataset. If you need to make minor changes, I recommend just copying the existing implementation and changing what you need to change. You are welcome to use (or override) the helper functions in the base class as needed.
3) When your file is run, set it up to call the functions in this order: `load_metadata`, `filter_metadata` (if applicable), `format_metadata_csv`, `update_data`. If you find you need to change this order or add additional functions for your specific dataset, you are welcome to do so.
4) In a Google Cloud VM, build the docker image (see below), then run the file you created to move your data to the `final` folder of the Google Cloud bucket.

To run your code:

1) Clone this repo in a GCP Virtual Machine (you are welcome to use `data-processor` as it already has many dependencies installed).
2) Build and run the docker image in interactive mode (see below).
3) Within the Docker image, manually run the Python file for your dataset to process the data.

### Main Functions in Workflow

`load_metadata`: The purpose of this function is to load the file containing your dataset's metadata from the bucket as a DataFrame for processing.

`filter_metadata`: The purpose of this function is to filter the metadata DataFrame from `load_metadata` to only include images you want to be used for the machine learning. You can decide what criteria to filter on, or not to filter at all if you so choose. For example, if your data have quality scores representing the quality of the labels, you can filter out images with a low quality score. Or you can filter out images taken with a tool other than a smartphone camera. Since everyone's filtering needs will differ, you must implement this function yourself based on your dataset's needs.

`format_metadata_csv`: The purpose of this function is to format your metadata CSV into a consistent format, so we can combine all of our core metadata (image IDs, labels, text descriptions, etc.) together into one file. Please see this function's docstring for the expected format of the CSV and be sure to adhere to that format. The function should return a DataFrame formatted and ready to be combined with `metadata_all.csv` (the .csv file containing the pooled metadata from all datasets).

* **Important**: Please do not add extra columns to `metadata_all.csv`. Your dataset's metadata unique to it should go in a separate CSV file in the `final` directory. It should not be combined with the pooled `metadata_all.csv`. The reason for this rule is to prevent the file from becoming too big and complex.

`update_data`: This function adds your data to the `metadata_all` CSV containing the pooled metadata from all datasets. If you implemented `format_metadata_csv` correctly, you should be able to just call the base class version of this function without modification. This function will add your images to the `metadata_all.csv` file in the `final` directory of the GCP bucket. It will not duplicate entries that already exist in that file, so if you run it multiple times on the same images, it will not duplicate those image entries. It will update the metadata for existing images if it is changed, however. This function also copies your images into the `final/imgs` folder and copies your dataset-specific metadata file to the `final` folder. It will override any image or metadata files that already exist there, as long as the filenames remain consistent.

Use the provided helper script:
```bash
./docker-shell.sh
```

### Notes
- The Dockerfile automatically runs all processors in sequence: `processor_ddi.py`, `processor_fitz.py`, `processor_skincap.py`, `processor_isic.py`, `processor_derm1m.py`, followed by `harmonize_labels_filter_desc.py`, and finally `verify_output.py`
- Ensure `secrets/` folder exists in the repository root
- The build command should be run from the `src/data-processor/` directory

## What Remains Untested

<table class="index" data-sortable>
    <thead>
        <tr class="tablehead" title="Click to sort">
            <th id="file" class="name" aria-sort="none" data-shortcut="f">File<span class="arrows"></span></th>
            <th class="spacer">&nbsp;</th>
            <th id="statements" aria-sort="none" data-default-sort-order="descending" data-shortcut="s">statements<span class="arrows"></span></th>
            <th id="missing" aria-sort="none" data-default-sort-order="descending" data-shortcut="m">missing<span class="arrows"></span></th>
            <th id="excluded" aria-sort="none" data-default-sort-order="descending" data-shortcut="x">excluded<span class="arrows"></span></th>
            <th class="spacer">&nbsp;</th>
            <th id="coverage" aria-sort="none" data-shortcut="c">coverage<span class="arrows"></span></th>
        </tr>
    </thead>
    <tbody>
        <tr class="region">
            <td class="name"><a href="processor_py.html">processor.py</a></td>
            <td class="spacer">&nbsp;</td>
            <td>138</td>
            <td>6</td>
            <td>0</td>
            <td class="spacer">&nbsp;</td>
            <td data-ratio="132 138">96%</td>
        </tr>
    </tbody>
</table>
