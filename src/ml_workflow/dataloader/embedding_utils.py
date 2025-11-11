try:
    from ..utils import logger
    from ..io_utils import file_exists, save_dataframe_to_parquet, load_parquet
    from ..constants import TEXT_DESC_COL, IMG_ID_COL, EMBEDDING_COL, MODELS
except ImportError:
    from utils import logger
    from io_utils import file_exists, save_dataframe_to_parquet, load_parquet
    from constants import TEXT_DESC_COL, IMG_ID_COL, EMBEDDING_COL, MODELS

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Literal
import pandas as pd
import os
import ast

class MissingEmbeddingError(ValueError):
    pass

_MODEL_CACHE = {}

def array_to_embeddings_list(arr: np.ndarray):
    """
    Converts a 2D NumPy array to a list of lists suitable for storage in a Pandas DataFrame.

    Args:
        arr: shape (N, D)

    Returns:
        List of lists, len = N, each list of length D
    """
    return arr.tolist()

def get_hf_cache_dir():
    # Look for environment variable first (recommended practice!)
    return os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool embeddings using last token"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def embedding_to_array(emb_val, dtype=np.float32):
    """
    Convert a DataFrame embedding value (string or list) to a NumPy array.

    Args:
        emb_val: List or string representing embedding.
        dtype: Numpy dtype.

    Returns:
        1D np.ndarray
    """
    if isinstance(emb_val, str):
        emb_val = ast.literal_eval(emb_val)
    return np.array(emb_val, dtype=dtype)

def mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool embeddings using mean pooling"""
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden_states * mask
    pooled = masked.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
    return pooled

def cls_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool embeddings using CLS token (first token, for SapBERT)"""
    return last_hidden_states[:, 0]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format text with task instruction (for QWEN models)"""
    return f'Instruct: {task_description}\nQuery: {query}'

@torch.no_grad()
def encode_with_transformers(model_name: str, texts: list[str], batch_size: int, max_length: int, 
                            device: torch.device, pooling_strategy: str = 'mean', qwen_instr: str="") -> np.ndarray:
    """Encode texts using transformers with specified pooling strategy

    If the text description is missing (null or empty string), this function replaces it with a [MISSING] token and generates a corresponding embedding.
    
    Args:
        model_name: HuggingFace model name
        texts: List of texts to encode
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        device: Device to use (cpu/cuda/mps)
        pooling_strategy: One of 'mean', 'cls', 'last_token'
        qwen_instr: The instructions for the QWEN model to use when generating an embedding (ignored if model is not QWEN).
    """
    # validate inputs
    if 'qwen' in model_name.lower():
        if pooling_strategy != 'last_token':
            raise ValueError("For QWEN models, only 'last_token' pooling is supported.")
    else:
        if pooling_strategy == 'last_token':
            raise ValueError("'last_token' pooling is only intended for QWEN models.")

    # replace nulls or empty strings with [MISSING] token
    texts = ["[MISSING]" if pd.isna(text) or text == "" else text for text in texts]

    # if a simple model name is passed in (e.g. qwen, pubmedbert), convert to full name
    if model_name in MODELS.keys():
        model_name = MODELS[model_name]

    cache_key = f"transformers_{model_name}"
    if cache_key not in _MODEL_CACHE:
        logger.info(f"Loading transformers model: {model_name}")
        cache_dir = get_hf_cache_dir()
        padding_side = 'left' if pooling_strategy == 'last_token' else 'right'
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
        tokenizer.padding_side = padding_side
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, use_safetensors=True).to(device).eval()
        _MODEL_CACHE[cache_key] = (tokenizer, model)
    
    tokenizer, model = _MODEL_CACHE[cache_key]
    
    if pooling_strategy == 'last_token':
        task_description = qwen_instr
        texts = [get_detailed_instruct(task_description, text) for text in texts]
    
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        try:
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            for k in enc:
                enc[k] = enc[k].to(device)
            out = model(**enc)
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            attention_mask = enc["attention_mask"]
            if pooling_strategy == 'last_token':
                pooled = last_token_pool(last_hidden, attention_mask)
            elif pooling_strategy == 'cls':
                pooled = cls_pool(last_hidden, attention_mask)
            elif pooling_strategy == 'mean':
                pooled = mean_pool(last_hidden, attention_mask)
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}. Choose from 'mean', 'cls', 'last_token'")
            embs.append(pooled.detach().cpu().numpy())
        except Exception as e:
            logger.error("Error encoding batch %d: %s. Replacing batch with all zeros.", i//batch_size, e)
            # append embeddings of all zeros if the batch fails, to ensure embeddings line up properly with images in the output
            embs.append(np.zeros((len(chunk), model.config.hidden_size), dtype=np.float32))
    
    if not embs:
        logger.error("All batches failed during encoding")
        raise ValueError("All batches failed during encoding")
    
    # sanity check that every text was given an embedding (even if it failed) to ensure embeddings correspond with the right images
    result = np.vstack(embs)
    assert result.shape[0] == len(texts), f"Shape mismatch: got {result.shape[0]}, expected {len(texts)}"
    return result

def compute_embeddings_and_save(data: pd.DataFrame, path: str,
                           model_name: str, batch_size: int, max_length: int, 
                           device: torch.device, pooling_strategy: str = 'mean', qwen_instr: str=""):
    """Pre-compute embeddings and write them to a file.
    
    Stores the embeddings in a .parquet file with columns: image_id, text_desc, embedding.

    If the file(s) already exist, will recompute the embeddings and override it/them. Gives a warning in this case.

    Args:
        data: A DataFrame with the columns `image_id` and `text_desc`, containing the image ID and text description, respectively.
        path: The path of the file to write the data to. Begin with gs:// if it is a path in GCP. Relative paths will be saved locally to disk.
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        device: Device to use (cpu/cuda/mps)
        pooling_strategy: One of 'mean', 'cls', 'last_token'
        qwen_instr: The instructions for the QWEN model to use when generating an embedding (ignored if model is not QWEN).
    Returns:
        The stored embeddings as a DataFrame with columns image_id, text_desc, embedding
    """
    
    # check if files already exist
    if file_exists(path):
        logger.warning("WARNING: file with embeddings already exists. Will be overwritten with new embeddings.")

    # compute the embeddings
    texts = data['text_desc'].to_list()
    embeddings = encode_with_transformers(model_name, texts, batch_size, max_length, device, pooling_strategy, qwen_instr)
    embd_lst = array_to_embeddings_list(arr=embeddings)

    # save embeddings
    out_data = data[[IMG_ID_COL, TEXT_DESC_COL]]
    out_data[EMBEDDING_COL] = embd_lst
    if not pd.isna(path):
        save_dataframe_to_parquet(path, out_data, index=False)

    return out_data

def load_embeddings(file: str, image_ids: list[str], text_descs: list[str], on_not_found: Literal['warn', 'error']='error'):
    """Load existing embeddings for a set of images or text descriptions.

    If a particular image id or text description is not found, this function gives an error. 
    It can alternatively be set to give a warning instead. (In this case, embeddings will not be 
    generated for image_ids not in the file, so this setting is dangerous to enable).

    This function returns embeddings for images with an image_id in img_ids OR a text description in text_descs.
    
    Args:
        file: The file from which to load embeddings (can be on local machine or in GCS bucket).
        image_ids: The image IDs for which to load embeddings. If left blank, loads all image_ids in the file
        text_descs: The text descriptions for which to load embeddings. If left blank, loads embeddings for all text descriptions. 
        on_not_found: One of `'warn'` or `'error'`. Whether to warn or error if an embedding is not found.
    Returns: 
        A DataFrame with the columns image_id, text_desc, embedding
    """
    # load embeddings
    embeddings = load_parquet(file)

    # check that all image_ids and text_descs are present in the file
    query_img_ids = set([id for id in image_ids if not pd.isna(id)])
    existing_img_ids = set(embeddings[IMG_ID_COL].dropna())
    if not query_img_ids.issubset(existing_img_ids):
        img_ids_not_found = [i for i in query_img_ids if i not in existing_img_ids]
        if on_not_found == 'warn':
            logger.warning(f"WARNING: The following image_ids were not found in the embedding file: {img_ids_not_found}")
        elif on_not_found == 'error':
            raise MissingEmbeddingError(f"The following image_ids were not found in the embedding file: {img_ids_not_found}")
        else:
            raise ValueError(f"Invalid setting for on_not_found: {on_not_found}")
        
    query_text_desc = set([desc for desc in text_descs if not pd.isna(desc)])
    existing_text_desc = set(embeddings[TEXT_DESC_COL].dropna())
    if not query_text_desc.issubset(existing_text_desc):
        text_desc_not_found = [i for i in query_text_desc if i not in existing_text_desc]
        if on_not_found == 'warn':
            logger.warning(f"WARNING: The following text_descs were not found in the embedding file: {text_desc_not_found}")
        elif on_not_found == 'error':
            raise MissingEmbeddingError(f"The following text_descs were not found in the embedding file: {text_desc_not_found}")
        else:
            raise ValueError(f"Invalid setting for on_not_found: {on_not_found}")
        
    # filter to only include rows where the image_id or text_desc was requested
    keep_flg = embeddings[IMG_ID_COL].isin(query_img_ids) | embeddings[TEXT_DESC_COL].isin(query_text_desc)
    embeddings_filt = embeddings[keep_flg]

    return embeddings_filt

def load_or_compute_embeddings(data: pd.DataFrame, path: str,
                               model_name: str, batch_size: int, max_length: int, 
                               device: torch.device, pooling_strategy: str = 'mean', qwen_instr: str=""):
    """This function handles the loading or generation of embeddings for text descriptions.

    If the pre-specified path exists, this function does the following:

    - checks if the file contains embeddings for all the provided descriptions
        - if it does, this function simply returns them
    - if not, this function recomputes all the embeddings for the current list of images and overrides the file with the newly computed embeddings. 
      It also returns the newly computed embeddings.

    NOTE: we recompute all embeddings in case the configuration of the embedder is different from when the file was first generated. This way, the 
    file has consistent embeddings.

    Args:
        data: A DataFrame with the columns `image_id` and `text_desc`, containing the image ID and text description, respectively.
        path: The path of the file to write the data to. Begin with gs:// if it is a path in GCP. Relative paths will be saved locally to disk.
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        device: Device to use (cpu/cuda/mps)
        pooling_strategy: One of 'mean', 'cls', 'last_token'
        qwen_instr: The instructions for the QWEN model to use when generating an embedding (ignored if model is not QWEN).
    Returns:
        The stored embeddings as a DataFrame with columns image_id, text_desc, embedding
    """

    # check if the file contains embeddings for all the provided descriptions...
    if file_exists(path):
        logger.info(f"Embedding file exists at {path}")
        try:
            embeddings = load_embeddings(
                file=path,
                image_ids=data[IMG_ID_COL].to_list(),
                text_descs=data[TEXT_DESC_COL].to_list(),
                on_not_found='error',
            )
            # ... if it does, return them
            logger.info("Successfully loaded existing embeddings.")
            return embeddings
        except MissingEmbeddingError:
            # ... if it does not, recompute all embeddings
            logger.warning("Failed to load image embeddings from file. Some embeddings do not exist. Recomputing embeddings from scratch.")
            return compute_embeddings_and_save(data, path, model_name, batch_size, max_length, device, pooling_strategy, qwen_instr)
    else:
        # ... if the file does not exist, compute all embeddings
        logger.warning(f"File at path [{path}] does not exist. Computing embeddings from scratch.")
        return compute_embeddings_and_save(data, path, model_name, batch_size, max_length, device, pooling_strategy, qwen_instr)
