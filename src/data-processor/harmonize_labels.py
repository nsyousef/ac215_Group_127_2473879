import os
import io
from google.cloud import storage
import pandas as pd


def load_table_from_gcs(storage_client, bucket_name, blob_path: str, **kwargs):
    """
    Load a table/DataFrame from Google Cloud Storage.
    
    Supports multiple file formats including CSV, Parquet, and Excel files.
    The function automatically detects the file format based on the file extension
    and uses the appropriate pandas read function.
    
    Args:
        storage_client: Google Cloud Storage client instance
        bucket_name (str): Name of the GCS bucket
        blob_path (str): Path to the file within the bucket
        **kwargs: Additional keyword arguments passed to the pandas read function
        
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame
        
    Raises:
        ValueError: If the file extension is not supported
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    file_data = blob.download_as_bytes()
    _, ext = os.path.splitext(blob_path.lower())
    if ext in ['.csv', '.txt']:
        read_func = pd.read_csv
    elif ext in ['.parquet']:
        read_func = pd.read_parquet
    elif ext in ['.xls', '.xlsx']:
        read_func = pd.read_excel
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    file_like = io.BytesIO(file_data)
    return read_func(file_like, **kwargs)


def write_table_to_gcs(storage_client, bucket_name, df: pd.DataFrame, blob_path: str, **kwargs) -> None:
    """
    Writes a Pandas DataFrame as a CSV file to Google Cloud Storage.

    Args:
        storage_client: Google Cloud Storage client
        bucket_name (str): Name of the GCS bucket
        df (pd.DataFrame): The Pandas DataFrame to write.
        blob_path (str): Path to the file within the bucket.
        **kwargs: Keyword arguments forwarded to the pandas to_csv function. E.g. 'sep', 'header', etc.
        
    Returns:
        None
    """
    # Convert DataFrame to CSV in memory
    csv_buffer = io.BytesIO()
    # Pandas to_csv supports file-like objects; ensure writing bytes (so set encoding)
    df.to_csv(csv_buffer, index=False, **kwargs)
    csv_buffer.seek(0)

    # Get blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Upload to GCS
    blob.upload_from_file(csv_buffer, content_type='text/csv')
    print(f"Successfully wrote DataFrame to gs://{bucket_name}/{blob_path}")


def harmonize_labels(metadata_all):
    """
    Harmonize labels from different datasets by binning similar conditions together.
    Maintains clinical specificity while reducing redundancy from different naming conventions.

    Args:
        metadata_all (pd.DataFrame): DataFrame containing a 'label' column with medical condition labels
        
    Returns:
        pd.DataFrame: Copy of the input DataFrame with harmonized labels
    """
    metadata_all_harmonized = metadata_all.copy()
    
    # Define label mappings for harmonization
    label_mappings = {
        # Melanoma variations (including capitalized versions)
        'melanoma-in-situ': 'melanoma',
        'melanoma-acral-lentiginous': 'melanoma',
        'nodular-melanoma-(nm)': 'melanoma',
        'Melanoma Invasive': 'melanoma',
        'Melanoma in situ': 'melanoma',
        'Melanoma, NOS': 'melanoma',
        'Melanoma metastasis': 'melanoma-metastatic',
        
        # Basal cell carcinoma variations
        'basal-cell-carcinoma': 'basal-cell-carcinoma',
        'basal-cell-carcinoma-superficial': 'basal-cell-carcinoma',
        'basal-cell-carcinoma-nodular': 'basal-cell-carcinoma',
        'basal cell carcinoma': 'basal-cell-carcinoma',
        'Basal cell carcinoma': 'basal-cell-carcinoma',
        
        # Squamous cell carcinoma variations
        'squamous-cell-carcinoma': 'squamous-cell-carcinoma',
        'squamous-cell-carcinoma-in-situ': 'squamous-cell-carcinoma',
        'squamous-cell-carcinoma-keratoacanthoma': 'squamous-cell-carcinoma',
        'squamous cell carcinoma': 'squamous-cell-carcinoma',
        'Squamous cell carcinoma, NOS': 'squamous-cell-carcinoma',
        'Squamous cell carcinoma in situ': 'squamous-cell-carcinoma',
        'Squamous cell carcinoma, Invasive': 'squamous-cell-carcinoma',
        'Keratoacanthoma': 'squamous-cell-carcinoma',
        
        # Mycosis fungoides variations
        'mycosis-fungoides': 'mycosis-fungoides',
        'mycosis fungoides': 'mycosis-fungoides',
        
        # Seborrheic keratosis variations
        'seborrheic-keratosis': 'seborrheic-keratosis',
        'seborrheic-keratosis-irritated': 'seborrheic-keratosis',
        'Seborrheic keratosis': 'seborrheic-keratosis',
        'Pigmented benign keratosis': 'seborrheic-keratosis',
        
        # Actinic keratosis variations
        'actinic-keratosis': 'actinic-keratosis',
        'actinic keratosis': 'actinic-keratosis',
        'Solar or actinic keratosis': 'actinic-keratosis',
        
        # Kaposi sarcoma variations
        'kaposi-sarcoma': 'kaposi-sarcoma',
        'kaposi sarcoma': 'kaposi-sarcoma',
        
        # Prurigo nodularis variations
        'prurigo-nodularis': 'prurigo-nodularis',
        'prurigo nodularis': 'prurigo-nodularis',
        
        # Eczema variations
        'eczema-spongiotic-dermatitis': 'eczema',
        'dyshidrotic eczema': 'eczema-dyshidrotic',
        
        # Nevi variations (grouping common benign nevi)
        'melanocytic-nevi': 'benign-nevus',
        'blue-nevus': 'benign-nevus',
        'congenital nevus': 'benign-nevus',
        'epidermal-nevus': 'benign-nevus',
        'epidermal nevus': 'benign-nevus',
        'Nevus': 'benign-nevus',
        'naevus comedonicus': 'benign-nevus',
        
        # Atypical/dysplastic nevi (keep separate due to malignant potential)
        'dysplastic-nevus': 'dysplastic-nevus',
        'pigmented-spindle-cell-nevus-of-reed': 'atypical-nevus',
        'atypical-spindle-cell-nevus-of-reed': 'atypical-nevus',
        'Atypical melanocytic neoplasm': 'atypical-nevus',
        'Atypical intraepithelial melanocytic proliferation': 'atypical-nevus',
        
        # Wart variations
        'verruca-vulgaris': 'wart',
        'Verruca': 'wart',
        
        # Pyogenic granuloma variations
        'pyogenic-granuloma': 'pyogenic-granuloma',
        'granuloma pyogenic': 'pyogenic-granuloma',
        'Pyogenic granuloma': 'pyogenic-granuloma',
        
        # Dermatofibroma variations
        'Dermatofibroma': 'dermatofibroma',
        
        # Cyst variations
        'epidermal-cyst': 'epidermal-cyst',
        'pilar cyst': 'pilar-cyst',
        'mucous cyst': 'mucous-cyst',
        'Trichilemmal or isthmic-catagen or pilar cyst': 'pilar-cyst',
        'Infundibular or epidermal cyst': 'epidermal-cyst',
        
        # Lentigo variations
        'solar-lentigo': 'solar-lentigo',
        'Solar lentigo': 'solar-lentigo',
        'acral-melanotic-macule': 'melanotic-macule',
        'Lentigo NOS': 'lentigo',
        'Ink-spot lentigo': 'lentigo',
        'Mucosal melanotic macule': 'melanotic-macule',
        
        # Hemangioma/vascular lesions
        'arteriovenous-hemangioma': 'hemangioma',
        'Hemangioma': 'hemangioma',
        'angioma': 'hemangioma',
        'Angiofibroma': 'angiofibroma',
        'Angiokeratoma': 'angiokeratoma',
        
        # Scar variations
        'Scar': 'scar',
        
        # Molluscum variations
        'molluscum-contagiosum': 'molluscum-contagiosum',
        'Molluscum': 'molluscum-contagiosum',
        
        # Xanthogranuloma variations
        'juvenile xanthogranuloma': 'xanthogranuloma',
        'Juvenile xanthogranuloma': 'xanthogranuloma',
        
        # Acanthoma variations
        'clear-cell-acanthoma': 'clear-cell-acanthoma',
        'Clear cell acanthoma': 'clear-cell-acanthoma',
        
        # Keratosis variations
        'Lichen planus like keratosis': 'lichenoid-keratosis',
        'benign-keratosis': 'benign-keratosis',
        
        # Polyp variations
        'Fibroepithelial polyp': 'fibroepithelial-polyp',
        
        # Other variations with different naming conventions
        'seborrheic dermatitis': 'seborrheic-dermatitis',
        'allergic contact dermatitis': 'contact-dermatitis',
        'factitial dermatitis': 'dermatitis-factitia',
        'perioral dermatitis': 'perioral-dermatitis',
        
        # Lupus variations
        'lupus erythematosus': 'lupus-erythematosus',
        'lupus subacute': 'lupus-erythematosus',
        
        # Granuloma variations
        'granuloma annulare': 'granuloma-annulare',
        'foreign-body-granuloma': 'granuloma-foreign-body',
        
        # Hyperplasia variations
        'Sebaceous hyperplasia': 'sebaceous-hyperplasia',
        
        # Other systematic mappings for space-to-hyphen standardization
        'lichen planus': 'lichen-planus',
        'lichen simplex': 'lichen-simplex',
        'lichen amyloidosis': 'lichen-amyloidosis',
        'pityriasis rosea': 'pityriasis-rosea',
        'pityriasis rubra pilaris': 'pityriasis-rubra-pilaris',
        'pityriasis lichenoides chronica': 'pityriasis-lichenoides-chronica',
        'erythema multiforme': 'erythema-multiforme',
        'erythema nodosum': 'erythema-nodosum',
        'erythema annulare centrifigum': 'erythema-annulare-centrifigum',
        'erythema elevatum diutinum': 'erythema-elevatum-diutinum',
        'keratosis pilaris': 'keratosis-pilaris',
        'necrobiosis lipoidica': 'necrobiosis-lipoidica',
        'acanthosis nigricans': 'acanthosis-nigricans',
        'ichthyosis vulgaris': 'ichthyosis-vulgaris',
        'langerhans cell histiocytosis': 'langerhans-cell-histiocytosis',
        'xeroderma pigmentosum': 'xeroderma-pigmentosum',
        'incontinentia pigmenti': 'incontinentia-pigmenti',
        'urticaria pigmentosa': 'urticaria-pigmentosa',
        'aplasia cutis': 'aplasia-cutis',
        'calcinosis cutis': 'calcinosis-cutis',
        'ehlers danlos syndrome': 'ehlers-danlos-syndrome',
        'tuberous sclerosis': 'tuberous-sclerosis',
        'hailey hailey disease': 'hailey-hailey-disease',
        'dariers disease': 'dariers-disease',
        'behcets disease': 'behcets-disease',
        'lyme disease': 'lyme-disease',
        'tick bite': 'tick-bite',
        'nematode infection': 'nematode-infection',
        'neutrophilic dermatoses': 'neutrophilic-dermatoses',
        'neurotic excoriations': 'neurotic-excoriations',
        'fixed eruptions': 'fixed-eruptions',
        'fordyce spots': 'fordyce-spots',
        'stasis edema': 'stasis-edema',
        'drug induced pigmentary changes': 'drug-induced-pigmentary-changes',
        'pediculosis lids': 'pediculosis-lids',
        'tinea pedis': 'tinea-pedis',
        'acral melanotic macule': 'melanotic-macule',
        'clear cell acanthoma': 'clear-cell-acanthoma',
        'acne cystic': 'acne-cystic',
        
        # Additional mappings for missed labels
        'Mastocytosis': 'mastocytosis',
        'Supernumerary nipple': 'supernumerary-nipple',
        'Hidradenoma': 'hidradenoma',
        'Trichoblastoma': 'trichoblastoma',
        'Porokeratosis': 'porokeratosis',
    }
    
    # Apply the mappings
    metadata_all_harmonized['label'] = metadata_all_harmonized['label'].map(
        lambda x: label_mappings.get(x, x)
    )
    
    return metadata_all_harmonized


def main():
    # Define file paths and GCS configuration
    metadata_path = 'final/metadata_all.csv'
    storage_client = storage.Client()
    bucket_name = "derma-datasets-2"
    
    # Load original metadata from GCS
    metadata_all = load_table_from_gcs(storage_client, bucket_name, metadata_path)
    
    # Harmonize the labels
    metadata_all_harmonized = harmonize_labels(metadata_all)
    
    # Optional: Analyze label distribution
    print('Original labels:')
    print(metadata_all['label'].unique())
    print(f"# of labels: {len(metadata_all['label'].unique())}\n")
    print('Harmonized labels:')
    print(metadata_all_harmonized['label'].unique())
    print(f"# of labels: {len(metadata_all_harmonized['label'].unique())}\n")
    metadata_disease_count = metadata_all_harmonized['label'].value_counts()
    metadata_low_representation = metadata_disease_count[metadata_disease_count < 5]
    print(f'{len(metadata_low_representation)} diseases have less than 5 images')
    print(metadata_low_representation)

    # Save harmonized metadata back to GCS
    harmonized_output_path = 'final/metadata_all_harmonized.csv'
    write_table_to_gcs(storage_client, bucket_name, metadata_all_harmonized, harmonized_output_path)


if __name__ == '__main__':
    main()
