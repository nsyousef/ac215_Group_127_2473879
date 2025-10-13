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
        'lentigo maligna': 'melanoma-in-situ',  # Pre-melanoma
        'malignant melanoma': 'melanoma',
        'superficial spreading melanoma ssm': 'melanoma',
        'Melanoma metastasis': 'melanoma-metastatic',
        
        # Basal cell carcinoma variations
        'basal-cell-carcinoma': 'basal-cell-carcinoma',
        'basal-cell-carcinoma-superficial': 'basal-cell-carcinoma',
        'basal-cell-carcinoma-nodular': 'basal-cell-carcinoma',
        'basal cell carcinoma': 'basal-cell-carcinoma',
        'Basal cell carcinoma': 'basal-cell-carcinoma',
        'basal cell carcinoma morpheiform': 'basal-cell-carcinoma',
        'solid cystic basal cell carcinoma': 'basal-cell-carcinoma',
        
        # Squamous cell carcinoma variations
        'squamous-cell-carcinoma': 'squamous-cell-carcinoma',
        'squamous-cell-carcinoma-in-situ': 'squamous-cell-carcinoma',
        'squamous-cell-carcinoma-keratoacanthoma': 'squamous-cell-carcinoma',
        'squamous cell carcinoma': 'squamous-cell-carcinoma',
        'Squamous cell carcinoma, NOS': 'squamous-cell-carcinoma',
        'Squamous cell carcinoma in situ': 'squamous-cell-carcinoma',
        'Squamous cell carcinoma, Invasive': 'squamous-cell-carcinoma',
        'Keratoacanthoma': 'squamous-cell-carcinoma',
        
        # Other carcinomas (keep separate due to different prognosis)
        'sebaceous-carcinoma': 'sebaceous-carcinoma',
        'metastatic-carcinoma': 'metastatic-carcinoma',
        
        # Lymphomas and hematologic malignancies
        'mycosis-fungoides': 'mycosis-fungoides',
        'mycosis fungoides': 'mycosis-fungoides',
        'subcutaneous-t-cell-lymphoma': 'cutaneous-lymphoma',
        'leukemia-cutis': 'leukemia-cutis',
        'blastic-plasmacytoid-dendritic-cell-neoplasm': 'hematologic-malignancy',
        
        # Seborrheic keratosis variations
        'seborrheic-keratosis': 'seborrheic-keratosis',
        'seborrheic-keratosis-irritated': 'seborrheic-keratosis',
        'Seborrheic keratosis': 'seborrheic-keratosis',
        'seborrheic keratosis': 'seborrheic-keratosis',
        'Pigmented benign keratosis': 'seborrheic-keratosis',
        
        # Actinic keratosis variations
        'actinic-keratosis': 'actinic-keratosis',
        'actinic keratosis': 'actinic-keratosis',
        'Solar or actinic keratosis': 'actinic-keratosis',
        
        # Other keratoses
        'benign-keratosis': 'benign-keratosis',
        'inverted-follicular-keratosis': 'benign-keratosis',
        'lichenoid-keratosis': 'lichenoid-keratosis',
        'Lichen planus like keratosis': 'lichenoid-keratosis',
        'focal-acral-hyperkeratosis': 'hyperkeratosis',
        
        # Porokeratosis variants
        'porokeratosis': 'porokeratosis',
        'Porokeratosis': 'porokeratosis',
        'porokeratosis of mibelli': 'porokeratosis',
        'porokeratosis actinic': 'porokeratosis',
        'disseminated actinic porokeratosis': 'porokeratosis',
        
        # Kaposi sarcoma variations
        'kaposi-sarcoma': 'kaposi-sarcoma',
        'kaposi sarcoma': 'kaposi-sarcoma',
        
        # Prurigo nodularis variations
        'prurigo-nodularis': 'prurigo-nodularis',
        'prurigo nodularis': 'prurigo-nodularis',
        
        # Eczema/dermatitis variations
        'eczema-spongiotic-dermatitis': 'eczema',
        'eczema': 'eczema',
        'dyshidrotic eczema': 'eczema-dyshidrotic',
        'seborrheic dermatitis': 'seborrheic-dermatitis',
        'allergic contact dermatitis': 'contact-dermatitis',
        'factitial dermatitis': 'dermatitis-factitia',
        'perioral dermatitis': 'perioral-dermatitis',
        'neurodermatitis': 'neurodermatitis',
        
        # Nevi variations (grouping common benign nevi)
        'melanocytic-nevi': 'benign-nevus',
        'blue-nevus': 'benign-nevus',
        'congenital nevus': 'benign-nevus',
        'epidermal-nevus': 'benign-nevus',
        'epidermal nevus': 'benign-nevus',
        'Nevus': 'benign-nevus',
        'naevus comedonicus': 'benign-nevus',
        'nevus-lipomatosus-superficialis': 'benign-nevus',
        'nevus sebaceous of jadassohn': 'nevus-sebaceous',
        'becker nevus': 'benign-nevus',
        'nevocytic nevus': 'benign-nevus',
        'halo nevus': 'benign-nevus',
        
        # Atypical/dysplastic nevi (keep separate due to malignant potential)
        'dysplastic-nevus': 'dysplastic-nevus',
        'pigmented-spindle-cell-nevus-of-reed': 'atypical-nevus',
        'atypical-spindle-cell-nevus-of-reed': 'atypical-nevus',
        'Atypical melanocytic neoplasm': 'atypical-nevus',
        'Atypical intraepithelial melanocytic proliferation': 'atypical-nevus',
        
        # Wart variations
        'verruca-vulgaris': 'wart',
        'wart': 'wart',
        'Verruca': 'wart',
        'condyloma-accuminatum': 'wart-genital',
        
        # Viral infections
        'molluscum-contagiosum': 'molluscum-contagiosum',
        'Molluscum': 'molluscum-contagiosum',
        
        # Pyogenic granuloma variations
        'pyogenic-granuloma': 'pyogenic-granuloma',
        'granuloma pyogenic': 'pyogenic-granuloma',
        'Pyogenic granuloma': 'pyogenic-granuloma',
        
        # Other granulomas
        'foreign-body-granuloma': 'granuloma-foreign-body',
        'granuloma annulare': 'granuloma-annulare',
        'xanthogranuloma': 'xanthogranuloma',
        'juvenile xanthogranuloma': 'xanthogranuloma',
        'Juvenile xanthogranuloma': 'xanthogranuloma',
        
        # Dermatofibroma variations
        'dermatofibroma': 'dermatofibroma',
        'Dermatofibroma': 'dermatofibroma',
        'acquired-digital-fibrokeratoma': 'fibrous-lesion',
        'fibrous-papule': 'fibrous-lesion',
        
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
        'glomangioma': 'vascular-tumor',
        'angioleiomyoma': 'vascular-tumor',
        'lymphangioma': 'vascular-malformation',
        'Angiofibroma': 'angiofibroma',
        'Angiokeratoma': 'angiokeratoma',
        'telangiectases': 'telangiectases',
        'port wine stain': 'vascular-malformation',
        
        # Lipomatous lesions
        'lipoma': 'lipoma',
        
        # Neural lesions
        'neurofibroma': 'neurofibroma',
        'neuroma': 'neuroma',
        'cellular-neurothekeoma': 'neural-tumor',
        'neurofibromatosis': 'neurofibromatosis',
        
        # Adnexal tumors (hair/sweat gland)
        'trichilemmoma': 'adnexal-tumor',
        'trichofolliculoma': 'adnexal-tumor',
        'syringocystadenoma-papilliferum': 'adnexal-tumor',
        'eccrine-poroma': 'adnexal-tumor',
        'chondroid-syringoma': 'adnexal-tumor',
        'Hidradenoma': 'adnexal-tumor',
        'Trichoblastoma': 'adnexal-tumor',
        'syringoma': 'adnexal-tumor',
        'pilomatricoma': 'adnexal-tumor',
        
        # Scar variations
        'scar': 'scar',
        'Scar': 'scar',
        'keloid': 'scar-keloid',
        'striae': 'striae',
        
        # Skin tags
        'acrochordon': 'skin-tag',
        'Fibroepithelial polyp': 'skin-tag',
        
        # Acne variations
        'acne-cystic': 'acne',
        'acne vulgaris': 'acne',
        'acne': 'acne',
        
        # Follicular conditions
        'folliculitis': 'folliculitis',
        'hidradenitis': 'hidradenitis-suppurativa',
        
        # Hyperplasia variations
        'Sebaceous hyperplasia': 'sebaceous-hyperplasia',
        'reactive-lymphoid-hyperplasia': 'lymphoid-hyperplasia',
        
        # Acanthoma variations
        'clear-cell-acanthoma': 'clear-cell-acanthoma',
        'Clear cell acanthoma': 'clear-cell-acanthoma',
        'verruciform-xanthoma': 'xanthoma',
        
        # Pigmentary disorders
        'hyperpigmentation': 'hyperpigmentation',
        'vitiligo': 'vitiligo',
        'drug induced pigmentary changes': 'drug-induced-pigmentation',
        
        # Trauma/injury
        'abrasions-ulcerations-and-physical-injuries': 'trauma',
        'hematoma': 'hematoma',
        'abscess': 'abscess',
        
        # Infections
        'onychomycosis': 'fungal-infection',
        'tinea-pedis': 'fungal-infection',
        'scabies': 'parasitic-infection',
        'myiasis': 'parasitic-infection',
        'tungiasis': 'parasitic-infection',
        'pediculosis lids': 'parasitic-infection',
        'nematode infection': 'parasitic-infection',
        'coccidioidomycosis': 'fungal-infection',
        'paronychia': 'bacterial-infection',
        
        # Inflammatory conditions
        'lymphocytic-infiltrations': 'inflammatory-infiltrate',
        'morphea': 'morphea',
        'scleroderma': 'scleroderma',
        'graft-vs-host-disease': 'graft-vs-host-disease',
        'sarcoidosis': 'sarcoidosis',
        
        # Autoimmune/connective tissue
        'lupus erythematosus': 'lupus-erythematosus',
        'lupus subacute': 'lupus-erythematosus',
        'dermatomyositis': 'dermatomyositis',
        
        # Lichen conditions
        'lichen planus': 'lichen-planus',
        'lichen simplex': 'lichen-simplex',
        'lichen amyloidosis': 'lichen-amyloidosis',
        
        # Pityriasis conditions
        'pityriasis rosea': 'pityriasis-rosea',
        'pityriasis rubra pilaris': 'pityriasis-rubra-pilaris',
        'pityriasis lichenoides chronica': 'pityriasis-lichenoides-chronica',
        
        # Erythema conditions
        'erythema multiforme': 'erythema-multiforme',
        'erythema nodosum': 'erythema-nodosum',
        'erythema annulare centrifigum': 'erythema-annulare-centrifigum',
        'erythema elevatum diutinum': 'erythema-elevatum-diutinum',
        
        # Keratosis variations
        'keratosis pilaris': 'keratosis-pilaris',
        
        # Metabolic/genetic conditions
        'necrobiosis lipoidica': 'necrobiosis-lipoidica',
        'acanthosis nigricans': 'acanthosis-nigricans',
        'ichthyosis vulgaris': 'ichthyosis-vulgaris',
        'xeroderma pigmentosum': 'xeroderma-pigmentosum',
        'incontinentia pigmenti': 'incontinentia-pigmenti',
        'ehlers danlos syndrome': 'ehlers-danlos-syndrome',
        'tuberous sclerosis': 'tuberous-sclerosis',
        'hailey hailey disease': 'hailey-hailey-disease',
        'dariers disease': 'dariers-disease',
        'epidermolysis bullosa': 'epidermolysis-bullosa',
        
        # Psoriasis variants
        'psoriasis': 'psoriasis',
        'pustular psoriasis': 'psoriasis-pustular',
        
        # Rosacea
        'rosacea': 'rosacea',
        'rhinophyma': 'rhinophyma',
        'cheilitis': 'cheilitis',
        
        # Urticaria
        'urticaria': 'urticaria',
        'urticaria pigmentosa': 'urticaria-pigmentosa',
        
        # Mastocytosis
        'Mastocytosis': 'mastocytosis',
        
        # Miscellaneous conditions
        'Supernumerary nipple': 'supernumerary-nipple',
        'aplasia cutis': 'aplasia-cutis',
        'calcinosis cutis': 'calcinosis-cutis',
        'stasis edema': 'stasis-edema',
        'livedo reticularis': 'livedo-reticularis',
        'mucinosis': 'mucinosis',
        'scleromyxedema': 'scleromyxedema',
        'milia': 'milia',
        'fordyce spots': 'fordyce-spots',
        'xanthomas': 'xanthoma',
        'porphyria': 'porphyria',
        'langerhans cell histiocytosis': 'langerhans-cell-histiocytosis',
        'photodermatoses': 'photodermatitis',
        'sun damaged skin': 'sun-damage',
        'acrodermatitis enteropathica': 'acrodermatitis-enteropathica',
        
        # Drug reactions
        'drug eruption': 'drug-reaction',
        'stevens johnson syndrome': 'stevens-johnson-syndrome',
        'fixed eruptions': 'fixed-drug-eruption',
        
        # Behavioral/neurotic
        'neurotic excoriations': 'neurotic-excoriations',
        'neutrophilic dermatoses': 'neutrophilic-dermatoses',
        
        # Vector-borne
        'tick bite': 'tick-bite',
        'lyme disease': 'lyme-disease',
        'behcets disease': 'behcets-disease',
        
        # Additional systematic mappings
        'papilomatosis confluentes and reticulate': 'papillomatosis-confluent-reticulate',

        # Vascular lesions consolidation
        'glomangioma': 'hemangioma',  # Both are benign vascular tumors
        'angioleiomyoma': 'hemangioma',  # Both are benign vascular tumors

        # Fibrous lesions consolidation  
        'acquired-digital-fibrokeratoma': 'dermatofibroma',  # Both are fibrous lesions
        'fibrous-papule': 'dermatofibroma',  # Both are fibrous lesions

        # Granuloma consolidation
        'foreign-body-granuloma': 'granuloma-annulare',  # Merge granulomas together

        # Hyperplasia consolidation
        'Sebaceous hyperplasia': 'benign-tumor',  # Create general benign tumor category
        'reactive-lymphoid-hyperplasia': 'benign-tumor',

        # Neural tumor consolidation
        'neuroma': 'neurofibroma',  # Both are neural lesions
        'cellular-neurothekeoma': 'neurofibroma',  # Both are neural lesions

        # Hematologic malignancy consolidation
        'leukemia-cutis': 'hematologic-malignancy',
        'subcutaneous-t-cell-lymphoma': 'hematologic-malignancy',  # Change from cutaneous-lymphoma

        # Infection consolidation
        'abscess': 'bacterial-infection',  # Abscess is typically bacterial

        # Trauma consolidation
        'hematoma': 'trauma',  # Hematoma is usually trauma-related

        # Scleroderma spectrum consolidation
        'morphea': 'scleroderma',  # Morphea is localized scleroderma

        # Keratosis consolidation
        'focal-acral-hyperkeratosis': 'benign-keratosis',  # Both are benign keratoses

        # Pigmentary disorder consolidation
        'hyperpigmentation': 'pigmentary-disorder',  # Create broader category
        'drug induced pigmentary changes': 'pigmentary-disorder',

        # Inflammatory consolidation
        'lymphocytic-infiltrations': 'inflammatory-dermatosis',  # Create broader inflammatory category
        'neutrophilic dermatoses': 'inflammatory-dermatosis',
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
