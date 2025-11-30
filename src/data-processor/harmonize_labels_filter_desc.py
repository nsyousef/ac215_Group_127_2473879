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
    if ext in [".csv", ".txt"]:
        read_func = pd.read_csv
    elif ext in [".parquet"]:
        read_func = pd.read_parquet
    elif ext in [".xls", ".xlsx"]:
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
    blob.upload_from_file(csv_buffer, content_type="text/csv")
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
        # MALIGNANT LESIONS - Keep these separate (critical for patient safety)
        # Melanoma variations (including capitalized versions)
        "melanoma-in-situ": "melanoma",
        "melanoma-acral-lentiginous": "melanoma",
        "nodular-melanoma-(nm)": "melanoma",
        "Melanoma Invasive": "melanoma",
        "Melanoma in situ": "melanoma",
        "Melanoma, NOS": "melanoma",
        "lentigo maligna": "melanoma-in-situ",  # Pre-melanoma
        "malignant melanoma": "melanoma",
        "superficial spreading melanoma ssm": "melanoma",
        "Melanoma metastasis": "melanoma-metastatic",
        # Basal cell carcinoma variations
        "basal-cell-carcinoma": "basal-cell-carcinoma",
        "basal-cell-carcinoma-superficial": "basal-cell-carcinoma",
        "basal-cell-carcinoma-nodular": "basal-cell-carcinoma",
        "basal cell carcinoma": "basal-cell-carcinoma",
        "Basal cell carcinoma": "basal-cell-carcinoma",
        "basal cell carcinoma morpheiform": "basal-cell-carcinoma",
        "solid cystic basal cell carcinoma": "basal-cell-carcinoma",
        # Squamous cell carcinoma variations
        "squamous-cell-carcinoma": "squamous-cell-carcinoma",
        "squamous-cell-carcinoma-in-situ": "squamous-cell-carcinoma",
        "squamous-cell-carcinoma-keratoacanthoma": "squamous-cell-carcinoma",
        "squamous cell carcinoma": "squamous-cell-carcinoma",
        "Squamous cell carcinoma, NOS": "squamous-cell-carcinoma",
        "Squamous cell carcinoma in situ": "squamous-cell-carcinoma",
        "Squamous cell carcinoma, Invasive": "squamous-cell-carcinoma",
        "Keratoacanthoma": "squamous-cell-carcinoma",
        "keratoacanthoma": "squamous-cell-carcinoma",
        # Other malignancies - consolidate by tissue type
        "sebaceous-carcinoma": "sebaceous-carcinoma",  # Keep specific - user needs to know
        "metastatic-carcinoma": "metastatic-carcinoma",
        "metastatic carcinoma": "metastatic-carcinoma",
        "merkel cell carcinoma": "merkel-cell-carcinoma",  # Keep specific - aggressive cancer
        # Hematologic malignancies - consolidate
        "mycosis-fungoides": "cutaneous-lymphoma",
        "mycosis fungoides": "cutaneous-lymphoma",
        "subcutaneous-t-cell-lymphoma": "cutaneous-lymphoma",
        "leukemia-cutis": "cutaneous-lymphoma",
        "blastic-plasmacytoid-dendritic-cell-neoplasm": "cutaneous-lymphoma",
        "cutaneous b-cell lymphoma": "cutaneous-lymphoma",
        # Kaposi sarcoma
        "kaposi-sarcoma": "kaposi-sarcoma",
        "kaposi sarcoma": "kaposi-sarcoma",
        "kaposi's sarcoma of skin": "kaposi-sarcoma",
        # PRE-MALIGNANT LESIONS - Keep separate (important for monitoring)
        # Actinic keratosis variations
        "actinic-keratosis": "actinic-keratosis",
        "actinic keratosis": "actinic-keratosis",
        "Solar or actinic keratosis": "actinic-keratosis",
        # Atypical/dysplastic nevi (keep separate due to malignant potential)
        "dysplastic-nevus": "dysplastic-nevus",
        "pigmented-spindle-cell-nevus-of-reed": "dysplastic-nevus",
        "atypical-spindle-cell-nevus-of-reed": "dysplastic-nevus",
        "Atypical melanocytic neoplasm": "dysplastic-nevus",
        "Atypical intraepithelial melanocytic proliferation": "dysplastic-nevus",
        # BENIGN NEOPLASMS - Safe to consolidate by tissue type
        # Seborrheic keratosis variations
        "seborrheic-keratosis": "seborrheic-keratosis",
        "seborrheic-keratosis-irritated": "seborrheic-keratosis",
        "Seborrheic keratosis": "seborrheic-keratosis",
        "seborrheic keratosis": "seborrheic-keratosis",
        "seborrheic keratoses": "seborrheic-keratosis",
        'irritated seborrheic keratosis (from "sk/isk")': "seborrheic-keratosis",
        "Pigmented benign keratosis": "seborrheic-keratosis",
        # Other benign keratoses - consolidate
        "benign-keratosis": "benign-keratosis",
        "inverted-follicular-keratosis": "benign-keratosis",
        "lichenoid-keratosis": "benign-keratosis",
        "Lichen planus like keratosis": "benign-keratosis",
        "focal-acral-hyperkeratosis": "benign-keratosis",
        "keratosis pilaris": "benign-keratosis",
        # Porokeratosis variants
        "porokeratosis": "porokeratosis",
        "Porokeratosis": "porokeratosis",
        "porokeratosis of mibelli": "porokeratosis",
        "porokeratosis actinic": "porokeratosis",
        "disseminated actinic porokeratosis": "porokeratosis",
        # Common benign nevi - consolidate (all have similar management)
        "melanocytic-nevi": "benign-nevus",
        "blue-nevus": "benign-nevus",
        "congenital nevus": "benign-nevus",
        "epidermal-nevus": "benign-nevus",
        "epidermal nevus": "benign-nevus",
        "Nevus": "benign-nevus",
        "naevus comedonicus": "benign-nevus",
        "nevus-lipomatosus-superficialis": "benign-nevus",
        "becker nevus": "benign-nevus",
        "nevocytic nevus": "benign-nevus",
        "halo nevus": "benign-nevus",
        "nevus sebaceous of jadassohn": "benign-nevus",
        # Fibrous lesions - consolidate (similar management)
        "dermatofibroma": "dermatofibroma",  # Most common - keep specific
        "Dermatofibroma": "dermatofibroma",
        "acquired-digital-fibrokeratoma": "dermatofibroma",  # Can group with dermatofibroma
        "fibrous-papule": "dermatofibroma",
        "digital fibroma": "dermatofibroma",
        "fibroma molle": "dermatofibroma",
        "angiofibroma": "dermatofibroma",
        "Angiofibroma": "dermatofibroma",
        # Neural tumors - consolidate
        "neurofibroma": "neurofibroma",  # Common, users recognize this
        "neuroma": "neurofibroma",  # Can group together
        "cellular-neurothekeoma": "neurofibroma",
        "rheumatoid nodule": "neurofibroma",  # Actually not neural, but rare
        # Adnexal tumors - consolidate (all hair/sweat gland derived)
        "trichilemmoma": "hair-follicle-tumor",  # More user-friendly
        "trichofolliculoma": "hair-follicle-tumor",
        "syringocystadenoma-papilliferum": "sweat-gland-tumor",
        "eccrine-poroma": "sweat-gland-tumor",
        "chondroid-syringoma": "sweat-gland-tumor",
        "Hidradenoma": "sweat-gland-tumor",
        "Trichoblastoma": "hair-follicle-tumor",
        "syringoma": "sweat-gland-tumor",
        "pilomatricoma": "hair-follicle-tumor",
        "poroma": "sweat-gland-tumor",
        # Vascular lesions - consolidate benign vascular tumors
        "arteriovenous-hemangioma": "vascular-tumor",
        "Hemangioma": "vascular-tumor",
        "angioma": "vascular-tumor",
        "glomangioma": "vascular-tumor",
        "angioleiomyoma": "vascular-tumor",
        "lymphangioma": "vascular-tumor",
        "port wine stain": "vascular-tumor",
        "Angiokeratoma": "vascular-tumor",
        "angiokeratoma": "vascular-tumor",
        "venous lake": "vascular-tumor",
        "telangiectases": "vascular-tumor",
        "spider veins": "vascular-tumor",
        "telangiectasia macularis eruptiva perstans": "vascular-tumor",
        "campbell de morgan spots": "vascular-tumor",
        # Cysts - consolidate by type
        "epidermal-cyst": "epidermal-cyst",
        "epidermoid cyst": "epidermal-cyst",
        "Infundibular or epidermal cyst": "epidermal-cyst",
        "pilar cyst": "pilar-cyst",
        "Trichilemmal or isthmic-catagen or pilar cyst": "pilar-cyst",
        "mucous cyst": "mucous-cyst",
        "mucocele": "mucous-cyst",
        "myxoid cyst": "mucous-cyst",
        "steatocystoma multiplex": "epidermal-cyst",
        # Skin tags and similar lesions
        "acrochordon": "skin-tag",
        "Fibroepithelial polyp": "skin-tag",
        "skin tag": "skin-tag",
        # Lipomatous lesions
        "lipoma": "lipoma",
        # Granulomas - consolidate by type
        "pyogenic-granuloma": "pyogenic-granuloma",
        "pyogenic granuloma": "pyogenic-granuloma",
        "granuloma pyogenic": "pyogenic-granuloma",
        "Pyogenic granuloma": "pyogenic-granuloma",
        "granulation tissue": "pyogenic-granuloma",
        "foreign-body-granuloma": "granuloma-annulare",  # Group common ones
        "granuloma annulare": "granuloma-annulare",  # Keep most common one
        "granuloma faciale": "granuloma-annulare",
        "actinic granuloma": "granuloma-annulare",
        "majocchi granuloma": "fungal-infection",  # Actually fungal-related
        "xanthogranuloma": "xanthogranuloma",
        "juvenile xanthogranuloma": "xanthogranuloma",
        "Juvenile xanthogranuloma": "xanthogranuloma",
        # INFLAMMATORY CONDITIONS - Consolidate by pathophysiology
        # Eczema/dermatitis - consolidate common types
        "eczema-spongiotic-dermatitis": "eczema-dermatitis",
        "eczema": "eczema-dermatitis",
        "atopic dermatitis": "eczema-dermatitis",
        "dyshidrotic eczema": "eczema-dermatitis",
        "nummular eczema": "eczema-dermatitis",
        "discoid eczema": "eczema-dermatitis",
        "hand eczema": "eczema-dermatitis",
        "xerotic eczema": "eczema-dermatitis",
        "infected eczema": "eczema-dermatitis",
        "stasis dermatitis": "eczema-dermatitis",
        "seborrheic dermatitis": "eczema-dermatitis",
        "neurodermatitis": "eczema-dermatitis",
        "chronic actinic dermatitis": "eczema-dermatitis",
        "acute dermatitis": "eczema-dermatitis",
        "autoimmune dermatitis": "eczema-dermatitis",
        "exfoliative dermatitis": "eczema-dermatitis",
        "exfoliative erythroderma": "eczema-dermatitis",
        "inflammatory dermatosis": "eczema-dermatitis",
        # Contact dermatitis
        "allergic contact dermatitis": "contact-dermatitis",
        "contact dermatitis": "contact-dermatitis",
        "irritant contact dermatitis": "contact-dermatitis",
        "contact purpura": "contact-dermatitis",
        # Specialized dermatitis
        "perioral dermatitis": "perioral-dermatitis",
        "factitial dermatitis": "factitial-dermatitis",
        "dermatitis herpetiformis": "dermatitis-herpetiformis",
        # Psoriasis variants
        "psoriasis": "psoriasis",
        "pustular psoriasis": "psoriasis",
        "scalp psoriasis": "psoriasis",
        "palmoplantar pustulosis": "psoriasis",
        # Acne variants
        "acne-cystic": "acne",
        "acne vulgaris": "acne",
        "acne": "acne",
        "steroid acne": "acne",
        "acne urticata": "acne",
        "acne keloidalis nuchae": "acne",
        # Rosacea spectrum
        "rosacea": "rosacea",
        "rhinophyma": "rosacea",
        # Follicular conditions
        "folliculitis": "folliculitis",
        "kerion": "folliculitis",
        "hidradenitis": "hidradenitis-suppurativa",
        "hidradenitis suppurativa": "hidradenitis-suppurativa",
        "fox-fordyce disease": "folliculitis",
        # Urticaria variants
        "urticaria": "urticaria",
        "urticaria pigmentosa": "urticaria",
        # Prurigo conditions
        "prurigo-nodularis": "prurigo",
        "prurigo nodularis": "prurigo",
        "prurigo": "prurigo",
        "prurigo pigmentosa": "prurigo",
        "prurigo of pregnancy": "prurigo",
        "pruritus ani": "prurigo",
        # Lichen conditions
        "lichen planus": "lichen-planus",
        "lichen simplex": "lichen-planus",
        "lichen amyloidosis": "lichen-planus",
        "lichen striatus": "lichen-planus",
        "lichen spinulosus": "lichen-planus",
        # Pityriasis conditions
        "pityriasis rosea": "pityriasis",
        "pityriasis rubra pilaris": "pityriasis",
        "pityriasis lichenoides chronica": "pityriasis",
        "pityriasis lichenoides et varioliformis acuta": "pityriasis",
        "pityriasis lichenoides": "pityriasis",
        # Erythema conditions
        "erythema multiforme": "erythema-reactive",
        "erythema nodosum": "erythema-reactive",
        "erythema annulare centrifigum": "erythema-reactive",
        "erythema annulare centrifugum": "erythema-reactive",
        "erythema elevatum diutinum": "erythema-reactive",
        "erythema dyschromicum perstans": "erythema-reactive",
        "erythema craquele": "erythema-reactive",
        "erythema gyratum repens": "erythema-reactive",
        "annular erythema": "erythema-reactive",
        "superficial gyrate erythema": "erythema-reactive",
        # AUTOIMMUNE/CONNECTIVE TISSUE - Consolidate related conditions
        # Lupus spectrum
        "lupus erythematosus": "lupus-erythematosus",
        "lupus subacute": "lupus-erythematosus",
        "cutaneous lupus": "lupus-erythematosus",
        # Scleroderma spectrum
        "scleroderma": "scleroderma-morphea",
        "morphea": "scleroderma-morphea",
        "scleromyxedema": "scleroderma-morphea",
        # Other autoimmune
        "dermatomyositis": "dermatomyositis",  # Keep - muscle involvement
        "graft-vs-host-disease": "graft-vs-host-disease",  # Keep - transplant complication
        "sarcoidosis": "sarcoidosis",  # Keep - systemic disease
        "cutaneous sarcoidosis": "sarcoidosis",  # Merge with sarcoidosis
        # INFECTIONS - Consolidate by organism type
        # Fungal infections
        "onychomycosis": "fungal-infection",
        "tinea-pedis": "fungal-infection",
        "tinea pedis": "fungal-infection",
        "tinea corporis": "fungal-infection",
        "tinea cruris": "fungal-infection",
        "tinea": "fungal-infection",
        "tinea versicolor": "fungal-infection",
        "tinea manus": "fungal-infection",
        "candidiasis": "fungal-infection",
        "coccidioidomycosis": "fungal-infection",
        "fungal dermatitis": "fungal-infection",
        # Bacterial infections
        "paronychia": "bacterial-infection",
        "abscess": "bacterial-infection",
        "cellulitis": "bacterial-infection",
        "impetigo": "bacterial-infection",
        "furuncle": "bacterial-infection",
        "staphylococcal scalded skin syndrome": "bacterial-infection",
        "local infection of wound": "bacterial-infection",
        "pyoderma": "bacterial-infection",
        "pyoderma gangrenosum": "bacterial-infection",
        "erosive pustular dermatosis of the scalp": "bacterial-infection",
        "pitted keratolysis": "bacterial-infection",
        "bacterial": "bacterial-infection",
        # Viral infections
        "molluscum-contagiosum": "viral-infection",
        "Molluscum": "viral-infection",
        "molluscum contagiosum": "viral-infection",
        "herpes zoster": "viral-infection",
        "herpes simplex virus": "viral-infection",
        "viral exanthem": "viral-infection",
        "varicella": "viral-infection",
        "hand foot and mouth disease": "viral-infection",
        "parvovirus b19 infection": "viral-infection",
        # Wart/HPV infections
        "verruca-vulgaris": "wart-hpv",
        "wart": "wart-hpv",
        "Verruca": "wart-hpv",
        "verruca vulgaris": "wart-hpv",
        "condyloma-accuminatum": "wart-hpv",
        "condyloma acuminatum": "wart-hpv",
        "flat wart": "wart-hpv",
        "skin diseases caused by warts": "wart-hpv",
        # Parasitic infections
        "scabies": "parasitic-infection",
        "myiasis": "parasitic-infection",
        "tungiasis": "parasitic-infection",
        "pediculosis lids": "parasitic-infection",
        "nematode infection": "parasitic-infection",
        "cutaneous larva migrans": "parasitic-infection",
        "sand-worm eruption": "parasitic-infection",
        "cutaneous leishmaniasis": "parasitic-infection",
        # Other infections
        "syphilis": "syphilis",  # Keep specific - STD implications
        "lyme disease": "lyme-disease",  # Keep specific - systemic disease
        "skin and soft tissue atypical mycobacterial infection": "atypical-mycobacterial-infection",
        # TRAUMA/INJURY - Consolidate
        "abrasions-ulcerations-and-physical-injuries": "trauma",
        "hematoma": "trauma",
        "wound/abrasion": "trauma",
        "animal bite - wound": "trauma",
        "burn of forearm": "trauma",
        "ulcer": "trauma",
        "insect bite": "trauma",
        "tick bite": "trauma",
        "poisoning by nematocyst": "trauma",
        # Scars and fibrosis
        "scar": "scar-fibrosis",
        "Scar": "scar-fibrosis",
        "keloid": "scar-fibrosis",
        "striae": "scar-fibrosis",
        "red stretch marks": "scar-fibrosis",
        # PIGMENTARY DISORDERS - Consolidate
        # Hyperpigmentation
        "hyperpigmentation": "hyperpigmentation",
        "drug induced pigmentary changes": "hyperpigmentation",
        "drug-induced pigmentary changes": "hyperpigmentation",
        "medication-induced cutaneous pigmentation": "hyperpigmentation",
        "melanin pigmentation due to exogenous substance": "hyperpigmentation",
        "riehl melanosis": "hyperpigmentation",
        "dermatosis papulosa nigra": "hyperpigmentation",
        # Hypopigmentation
        "vitiligo": "hypopigmentation",
        "post-inflammatory hypopigmentation": "hypopigmentation",
        "idiopathic guttate hypomelanosis": "hypopigmentation",
        # Lentigo/melanotic macules - consolidate
        "solar-lentigo": "lentigo-solar",
        "Solar lentigo": "lentigo-solar",
        "sun spots": "lentigo-solar",
        "acral-melanotic-macule": "melanotic-macule",
        "Lentigo NOS": "melanotic-macule",
        "Ink-spot lentigo": "melanotic-macule",
        "Mucosal melanotic macule": "melanotic-macule",
        "mucosal melanotic macule": "melanotic-macule",
        "acral melanotic macule": "melanotic-macule",
        "café au lait macule": "melanotic-macule",
        # HYPERPLASIA/HAMARTOMAS
        "Sebaceous hyperplasia": "benign-hyperplasia",
        "reactive-lymphoid-hyperplasia": "benign-hyperplasia",
        "Supernumerary nipple": "benign-hyperplasia",
        # Clear cell lesions
        "clear-cell-acanthoma": "clear-cell-acanthoma",
        "Clear cell acanthoma": "clear-cell-acanthoma",
        # Xanthomatous lesions
        "verruciform-xanthoma": "xanthoma",
        "xanthomas": "xanthoma",
        "eruptive xanthoma": "xanthoma",
        "xanthelasma": "xanthoma",
        "diffuse xanthoma": "xanthoma",
        # METABOLIC/GENETIC CONDITIONS - Keep separate due to systemic implications
        # Genodermatoses
        "necrobiosis lipoidica": "genodermatosis",
        "acanthosis nigricans": "genodermatosis",
        "ichthyosis vulgaris": "genodermatosis",
        "ichthyosis": "genodermatosis",
        "xeroderma pigmentosum": "genodermatosis",
        "incontinentia pigmenti": "genodermatosis",
        "ehlers danlos syndrome": "genodermatosis",
        "tuberous sclerosis": "genodermatosis",
        "hailey hailey disease": "genodermatosis",
        "dariers disease": "genodermatosis",
        "epidermolysis bullosa": "genodermatosis",
        "neurofibromatosis": "genodermatosis",
        "aplasia cutis": "genodermatosis",
        "hereditary": "genodermatosis",
        # Metabolic deposits
        "mucinosis": "metabolic-deposit",
        "calcinosis cutis": "metabolic-deposit",
        "amyloidosis": "metabolic-deposit",
        # MASTOCYTOSIS/HISTIOCYTOSIS
        "Mastocytosis": "mastocytosis-histiocytosis",
        "langerhans cell histiocytosis": "mastocytosis-histiocytosis",
        # DRUG REACTIONS - Consolidate
        "drug eruption": "drug-reaction",
        "stevens johnson syndrome": "drug-reaction-severe",
        "stevens-johnson syndrome": "drug-reaction-severe",
        "fixed eruptions": "drug-reaction",
        "fixed drug eruption": "drug-reaction",
        "acute generalized exanthematous pustulosis": "drug-reaction-severe",
        # BULLOUS DISEASES
        "bullous disease": "bullous-disease",
        "bullous pemphigoid": "bullous-disease",
        "childhood bullous pemphigoid": "bullous-disease",
        "pemphigus vulgaris": "bullous-disease",
        "acquired autoimmune bullous diseaseherpes gestationis": "bullous-disease",
        # PHOTODERMATOLOGY
        "photodermatoses": "photodermatitis",
        "sun damaged skin": "photodermatitis",
        "sunburn": "photodermatitis",
        "phytophotodermatitis": "photodermatitis",
        "polymorphous light eruption": "photodermatitis",
        "actinic solar damage(telangiectasia)": "photodermatitis",
        "actinic solar damage(solar purpura)": "photodermatitis",
        "actinic solar damage(solar elastosis)": "photodermatitis",
        "actinic solar damage(cutis rhomboidalis nuchae)": "photodermatitis",
        "radiodermatitis": "photodermatitis",
        "polymorphic eruption of pregnancy": "photodermatitis",
        # NAIL DISORDERS
        "onycholysis": "nail-disorder",
        "onychoschizia": "nail-disorder",
        "leukonychia": "nail-disorder",
        "koilonychia": "nail-disorder",
        "beau's lines": "nail-disorder",
        # MISCELLANEOUS BENIGN CONDITIONS
        # Milia and fordyce spots
        "milia": "milia-fordyce",
        "fordyce spots": "milia-fordyce",
        # Edema/circulatory
        "stasis edema": "circulatory-disorder",
        "livedo reticularis": "circulatory-disorder",
        "elephantiasis nostras": "circulatory-disorder",
        "poikiloderma": "circulatory-disorder",
        "poikiloderma of civatte": "circulatory-disorder",
        "flushing": "circulatory-disorder",
        # Keratotic conditions
        "hyperkeratosis palmaris et plantaris": "hyperkeratosis",
        "callus": "hyperkeratosis",
        "knuckle pads": "hyperkeratosis",
        "juvenile plantar dermatosis": "hyperkeratosis",
        "keratolysis exfoliativa of wende": "hyperkeratosis",
        "cutaneous horn": "hyperkeratosis",
        "atopic winter feet": "hyperkeratosis",
        # Desquamation/xerosis
        "xerosis": "xerosis-desquamation",
        "desquamation": "xerosis-desquamation",
        # Cheilitis
        "cheilitis": "cheilitis",
        # Geographic conditions
        "geographic tongue": "geographic-dermatosis",
        # Papillomatosis
        "papilomatosis confluentes and reticulate": "papillomatosis",
        "confluent and reticulated papillomatosis": "papillomatosis",
        # Pregnancy-related dermatoses
        "papular dermatoses of pregnancy": "pregnancy-dermatosis",
        "cholestasis of pregnancy": "pregnancy-dermatosis",
        # Intertrigo
        "intertrigo": "intertrigo",
        # Mucositis
        "muzzle rash": "mucositis",
        # Hair disorders
        "alopecia areata": "hair-disorder",
        "hair diseases": "hair-disorder",
        "hypertrichosis": "hair-disorder",
        # Miscellaneous rare conditions
        "chilblain": "environmental-dermatosis",
        "degos disease": "rare-dermatosis",
        "relapsing polychondritis": "rare-dermatosis",
        "behcets disease": "rare-dermatosis",
        # Neurotic/behavioral conditions
        "neurotic excoriations": "neurotic-dermatoses",
        "skin lesion in drug addict": "neurotic-dermatoses",
        # Dilated pore conditions
        "dilated pore of winer": "follicular-disorder",
        # Clubbing
        "clubbing of fingers": "nail-disorder",
        # Crowe's sign (neurofibromatosis)
        "crowe's sign": "genodermatosis",
        # Unilateral dermatoses
        "unilateral laterothoracic exanthem": "rare-dermatosis",
        # Proliferative conditions
        "proliferations": "benign-hyperplasia",
        # Lymphocytic infiltrate
        "lymphocytic infiltrate of jessner": "inflammatory-infiltrate",
        # Additional miscellaneous conditions
        "porphyria": "metabolic-disorder",
        "eruptive odontogenic cyst": "rare-cyst",
        "chondrodermatitis nodularis helicis": "chondrodermatitis",
        # Neutrophilic dermatoses (formatting fix)
        "neutrophilic dermatoses": "neutrophilic-dermatoses",
        # Acrodermatitis (formatting fix)
        "acrodermatitis enteropathica": "acrodermatitis-enteropathica",
    }

    # Apply the mappings
    metadata_all_harmonized["label"] = metadata_all_harmonized["label"].map(lambda x: label_mappings.get(x, x))

    return metadata_all_harmonized


def remove_diagnoses_from_descriptions(metadata_all_harmonized):
    """
    OPTIMIZED & BULLETPROOF version - Remove ALL diagnosis-referencing sentences while preserving clinical info.
    """
    import re
    import pandas as pd
    from functools import lru_cache
    
    # Build disease terms
    disease_terms = set()
    unique_labels = metadata_all_harmonized["label"].dropna().unique()

    for label in unique_labels:
        disease_terms.add(label.lower())
        parts = label.lower().replace("-", " ").replace("_", " ")
        disease_terms.add(parts)
        words = parts.split()
        if len(words) >= 2:
            disease_terms.add(parts)
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    phrase = " ".join(words[i:j])
                    if len(phrase) > 4:
                        disease_terms.add(phrase)

    additional_medical_terms = {
        "melanoma", "carcinoma", "lymphoma", "sarcoma", "keratosis",
        "dermatitis", "eczema", "psoriasis", "nevus", "nevi", "neoplasm",
        "syndrome", "basal cell", "squamous cell", "actinic", "seborrheic",
        "atopic", "tinea", "candidiasis", "vitiligo", "rosacea", "lupus",
        "scabies", "impetigo", "cellulitis", "abscess", "folliculitis",
        "herpes", "zoster", "shingles", "chickenpox", "varicella",
        "molluscum", "warts", "verruca", "acne", "urticaria", "hives",
        "angioedema", "pemphigus", "bullous", "erythema", "granuloma",
        "scleroderma", "morphea", "lichen", "pityriasis", "ichthyosis",
        "xanthoma", "lipoma", "hemangioma", "pyoderma",
        # Additional terms to catch remaining leakage
        "pemphigoid", "keloid", "fibroma", "papilloma", "atheroma",
        "neurofibroma", "trichoepithelioma", "dermographism", "behcet",
        "necrobiosis", "pustulosis", "pseudoporphyria", "rhinoscleroma",
        "poikiloderma", "telangiectasia", "angiofibroma", "syringoma",
    }
    disease_terms.update(additional_medical_terms)
    overly_generic = {"disease", "disorder", "condition", "skin", "rash", "lesion"}
    disease_terms = disease_terms - overly_generic

    # Filter out very short terms and sort by length (longest first for better matching)
    disease_terms = [term for term in disease_terms if len(term) >= 4]
    disease_terms = sorted(disease_terms, key=len, reverse=True)
    
    print(f"Loaded {len(disease_terms)} disease terms for filtering")

    # =================================================================
    # PRE-COMPILE ALL REGEX PATTERNS (huge performance boost)
    # =================================================================
    
    # Create a single combined regex for disease terms
    disease_pattern = r"\b(" + "|".join(re.escape(term) for term in disease_terms) + r")\b"
    disease_regex = re.compile(disease_pattern, re.IGNORECASE)
    
    # Catch words with disease-indicating suffixes
    medical_suffix_regex = re.compile(
        r'\b\w+(oma|itis|osis|derma|plasia|trophy|pathy|cele|ectomy|plasty)\b',
        re.IGNORECASE
    )
    
    # Pre-compile all other patterns
    pure_demographic_regexes = [
        re.compile(r"^the sex is (male|female)\.?$", re.IGNORECASE),
        re.compile(r"^the (approximate )?age is \d+\.?$", re.IGNORECASE),
        re.compile(r"^the gender is (male|female)\.?$", re.IGNORECASE),
        re.compile(r"^the body location is [\w\s,/]+\.?$", re.IGNORECASE),
        re.compile(r"^the symptoms are [\w\s,]+\.?$", re.IGNORECASE),
        re.compile(r"^the duration (of the skin condition )?is [\w\s]+\.?$", re.IGNORECASE),
        re.compile(r"^(male|female), \d+ years? old\.?$", re.IGNORECASE),
    ]
    
    disease_start_regex = re.compile(
        r"^(" + "|".join(re.escape(term) for term in disease_terms[:100]) + r")[\s:,]",  # Top 100 most common
        re.IGNORECASE
    )
    
    explicit_diagnosis_regexes = [
        re.compile(r"\bdiagnosed (as|with)\b", re.IGNORECASE),
        re.compile(r"\bdiagnosis (is|was|of|:)\s", re.IGNORECASE),
        re.compile(r"\b(final|clinical|confirmed|differential|initial|primary|secondary) diagnosis\b", re.IGNORECASE),
        re.compile(r"\bpatholog(ical|y) (confirmed|diagnosis)\b", re.IGNORECASE),
        re.compile(r"\bconsensus (is|was|diagnosis)\b", re.IGNORECASE),
        re.compile(r"\bidentified as\b", re.IGNORECASE),
        re.compile(r"\bconfirmed (as|to be)\b", re.IGNORECASE),
        re.compile(r"\b(seems|appears) (like|to be)\b", re.IGNORECASE),
        re.compile(r"\bmost likely\b", re.IGNORECASE),
        re.compile(r"\bprobably\b.*\b(is|has)\b", re.IGNORECASE),
        re.compile(r"\blikely (is|has|diagnosis)\b", re.IGNORECASE),
        re.compile(r"\bindicative of\b", re.IGNORECASE),
        re.compile(r"\btypical (of|for|presentation)\b", re.IGNORECASE),
        re.compile(r"\bcharacteristic (of|for)\b", re.IGNORECASE),
        re.compile(r"\bsuggestive of\b", re.IGNORECASE),
        re.compile(r"\bcompatible with\b", re.IGNORECASE),
        re.compile(r"\bconsistent with\b", re.IGNORECASE),
        re.compile(r"\bpathognomonic\b", re.IGNORECASE),
    ]
    
    image_description_regexes = [
        re.compile(r"^this (image|photo|picture|figure)", re.IGNORECASE),
        re.compile(r"^the (image|photo|picture|figure)", re.IGNORECASE),
        re.compile(r"^(clinical|dermoscopic) (image|photo|picture)", re.IGNORECASE),
        re.compile(r"^(representative|shown|depicting)", re.IGNORECASE),
    ]
    
    this_is_regex = re.compile(
        r"^(this|these|it) (is|are|was|were|shows?|depicts?|describes?|represents?|displays?|features?|highlights?|illustrates?|discusses?)\s",
        re.IGNORECASE
    )
    
    forum_regexes = [
        re.compile(r"(users?|members?|participants?|forum|discussion|replies?|consensus) (suggest|suggested|recommend|indicated|confirmed|agreed|noted|mentioned|pointed out|discussed)", re.IGNORECASE),
        re.compile(r"(several|many|some|multiple) users", re.IGNORECASE),
        re.compile(r"(the )?(primary|predominant|main) (opinion|diagnosis|suggestion)", re.IGNORECASE),
        re.compile(r"(most|majority of) (users?|participants?)", re.IGNORECASE),
        re.compile(r"discussion (among|suggests?|indicates?)", re.IGNORECASE),
    ]
    
    disease_statement_regexes = [
        re.compile(r"\b(can|may|might|could|will|is|are|was|were) (be|cause|lead|result|present|appear)", re.IGNORECASE),
        re.compile(r"\b(affects?|involves?|presents?)\b", re.IGNORECASE),
        re.compile(r"\b(caused by|due to|related to|associated with)\b", re.IGNORECASE),
        re.compile(r"\b(treatment|therapy) (for|of|includes?|options?)\b", re.IGNORECASE),
    ]
    
    diagnostic_jargon_regexes = [
        re.compile(r"^characterized by\b", re.IGNORECASE),
        re.compile(r"^features include\b", re.IGNORECASE),
        re.compile(r"^presents? with\b", re.IGNORECASE),
        re.compile(r"^(clinical|typical) (features?|presentation|appearance|findings?)\b", re.IGNORECASE),
        re.compile(r"^manifestations? (of|include)\b", re.IGNORECASE),
        re.compile(r"^(the )?(condition|disease) (is|can be|may be)\b", re.IGNORECASE),
        re.compile(r"^(the )?(cause|etiology|pathogenesis)\b", re.IGNORECASE),
    ]
    
    diagnosis_condition_regex = re.compile(r"(diagnosis|condition|disease)", re.IGNORECASE)

    # =================================================================
    # OPTIMIZED CHECKING FUNCTIONS
    # =================================================================
    
    def contains_disease_term(sentence_lower):
        """Fast disease term detection using pre-compiled regex."""
        return disease_regex.search(sentence_lower) is not None
    
    def is_pure_demographic_or_observation(sentence_lower):
        """Check if sentence is purely demographic."""
        if len(sentence_lower) < 10:
            return False
        return any(regex.match(sentence_lower) for regex in pure_demographic_regexes)
    
    def is_diagnosis_sentence(sentence):
        """Optimized diagnosis detection with pre-compiled patterns."""
        sentence_lower = sentence.strip().lower()
        
        if len(sentence_lower) < 3:
            return False
        
        word_count = len(sentence_lower.split())
    
        # =================================================================
        # STEP 1: Remove very short sentences (1-5 words)
        # =================================================================
        if word_count <= 5:
            # Contains disease term
            if contains_disease_term(sentence_lower):
                return True
            
            # Contains medical suffix (likely a disease name)
            if medical_suffix_regex.search(sentence_lower):
                return True

            # Single medical word (catches standalone terms like "furuncle")
            if word_count == 1 and len(sentence_lower) > 5:
                return True
            
            # Is a standalone capitalized medical term (like a label)
            if sentence.strip().endswith('.') and sentence.strip()[0].isupper() and word_count <= 3:
                return True
        
        # =================================================================
        # STEP 2: Remove 6-20 word sentences that are mostly disease names
        # =================================================================
        if 6 <= word_count <= 20:
            if contains_disease_term(sentence_lower):
                # Count meaningful content words (not stopwords)
                sentence_without_stopwords = re.sub(
                    r'\b(the|body|location|is|are|was|were|on|in|at|of|and|or|with|to|a|an|for|by|from)\b',
                    '', sentence_lower
                )
                # If <40 chars of real content, it's mostly a disease name
                if len(sentence_without_stopwords.strip()) < 40:
                    return True
        
        # =================================================================
        # STEP 3: Keep pure demographics
        # =================================================================
        if is_pure_demographic_or_observation(sentence_lower):
            return False
        
        # =================================================================
        # STEP 4: Remove sentences starting with disease terms
        # =================================================================
        if disease_start_regex.match(sentence_lower):
            return True
        
        # =================================================================
        # STEP 5: Explicit diagnosis patterns
        # =================================================================
        if any(regex.search(sentence_lower) for regex in explicit_diagnosis_regexes):
            return True
        
        # =================================================================
        # STEP 6: Image descriptions
        # =================================================================
        if any(regex.search(sentence_lower) for regex in image_description_regexes):
            return True
        
        # =================================================================
        # STEP 7: "This is/shows" statements
        # =================================================================
        if this_is_regex.search(sentence_lower):
            if contains_disease_term(sentence_lower) or len(sentence_lower) < 150:
                return True
        
        # =================================================================
        # STEP 8: Forum patterns
        # =================================================================
        for regex in forum_regexes:
            if regex.search(sentence_lower):
                if diagnosis_condition_regex.search(sentence_lower) or contains_disease_term(sentence_lower):
                    return True
        
        # =================================================================
        # STEP 9: Disease terms in sentence
        # =================================================================
        has_disease_term = contains_disease_term(sentence_lower)
        
        if has_disease_term:
            # Disease statement patterns
            if any(regex.search(sentence_lower) for regex in disease_statement_regexes):
                return True
            
            # Length-based filtering
            if len(sentence_lower) < 200:
                return True
            
            # Position-based filtering for longer sentences
            first_portion = sentence_lower[:int(len(sentence_lower) * 0.3)]
            last_portion = sentence_lower[int(len(sentence_lower) * 0.7):]
            
            if disease_regex.search(first_portion) or disease_regex.search(last_portion):
                return True
        
        # =================================================================
        # STEP 10: Diagnostic jargon
        # =================================================================
        if any(regex.search(sentence_lower) for regex in diagnostic_jargon_regexes):
            return True
        
        # =================================================================
        # STEP 11: Final catch-all for standalone medical terms
        # =================================================================
        if word_count <= 10 and sentence.strip().endswith('.'):
            if medical_suffix_regex.search(sentence_lower) or contains_disease_term(sentence_lower):
                # But NOT if it's describing patient symptoms/experiences
                if not re.search(r'\b(patient|person|individual|poster|user|reports?|experiencing|presents?|symptoms?)\b', sentence_lower):
                    return True
        
        return False

    def clean_description(text):
        """Clean a single text description."""
        if pd.isna(text) or text == "":
            return text
        
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", str(text))
        
        # Filter sentences
        cleaned_sentences = [
            sentence for sentence in sentences
            if sentence and sentence.strip() and not is_diagnosis_sentence(sentence)
        ]
        
        cleaned_text = " ".join(cleaned_sentences).strip()
        return cleaned_text if cleaned_text else None

    # =================================================================
    # PROCESS DATA
    # =================================================================
    
    df_cleaned = metadata_all_harmonized.copy()
    
    if "text_desc" in df_cleaned.columns:
        print("Removing diagnosis-referencing sentences from text descriptions...")
        
        # Use pandas apply (can be parallelized if needed)
        df_cleaned["text_desc"] = df_cleaned["text_desc"].apply(clean_description)
        
        # Statistics
        original_lengths = metadata_all_harmonized["text_desc"].fillna("").str.len()
        new_lengths = df_cleaned["text_desc"].fillna("").str.len()
        modified_count = (original_lengths != new_lengths).sum()
        completely_removed = (new_lengths == 0).sum()
        
        print(f"Modified {modified_count} out of {len(df_cleaned)} descriptions")
        print(f"Completely removed: {completely_removed} descriptions")
        print(f"Removed avg {(original_lengths - new_lengths).mean():.1f} chars per description")
        print(f"Percentage removed: {(completely_removed/len(df_cleaned)*100):.1f}%")
    else:
        print("Warning: 'text_desc' column not found")
    
    return df_cleaned


def main():
    # Define file paths and GCS configuration
    metadata_path = "final/metadata_all.csv"
    storage_client = storage.Client()
    bucket_name = "derma-datasets-2"

    # Load original metadata from GCS
    metadata_all = load_table_from_gcs(storage_client, bucket_name, metadata_path)

    # Harmonize the labels
    metadata_all_harmonized = harmonize_labels(metadata_all)

    # Optional: Analyze label distribution
    print("Original labels:")
    print(metadata_all["label"].unique())
    print(f"# of labels: {len(metadata_all['label'].unique())}\n")
    print("Harmonized labels:")
    print(metadata_all_harmonized["label"].unique())
    print(f"# of labels: {len(metadata_all_harmonized['label'].unique())}\n")
    metadata_disease_count = metadata_all_harmonized["label"].value_counts()
    metadata_low_representation = metadata_disease_count[metadata_disease_count < 5]
    print(f"{len(metadata_low_representation)} diseases have less than 5 images")
    print(metadata_low_representation)

    # Remove diagnoses from descriptions
    metadata_cleaned = remove_diagnoses_from_descriptions(metadata_all_harmonized)

    harmonized_output_path = "final/metadata_all_harmonized.csv"
    write_table_to_gcs(storage_client, bucket_name, metadata_cleaned, harmonized_output_path)


if __name__ == "__main__":
    main()
