"""
embedding.py
------------
Phase 2: Sentence-BERT Feature Extraction and Embedding Precomputation
Project: Intelligent Question Quality Assessment and Expert Routing System for Q&A Platforms

This module:
1. Loads the pretrained Sentence-BERT model 'all-MiniLM-L6-v2' (producing 384-dimensional vectors).
2. Encodes the cleaned question texts for Train, Validation, and Test splits.
3. Normalizes embeddings (L2 unit norm) so dot products directly equate to cosine similarity.
4. Serializes embeddings and integer labels to data/embeddings/ for fast MLP training and reuse.
5. Provides a standalone SentenceEncoder class for real-time inference in subsequent phases.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import logging
from typing import List, Union, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ground truth label mapping (reproducible and explicit)
LABEL_TO_INT: Dict[str, int] = {
    "HQ": 0,
    "LQ_EDIT": 1,
    "LQ_CLOSE": 2
}
INT_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


class SentenceEncoder:
    """
    Wrapper around Sentence-BERT (all-MiniLM-L6-v2) for feature extraction.
    
    Academic Rationale:
    'all-MiniLM-L6-v2' is a distilled Transformer encoder fine-tuned on >1 billion sentence pairs
    using self-supervised contrastive learning. It maps variable-length texts into a 384-dimensional
    dense semantic vector space where semantically similar texts are clustered together.
    It provides 5x faster inference than BERT-base while retaining ~99% of its semantic performance,
    making it ideal for academic deployment on consumer hardware.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading Sentence-BERT model '{model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = 384
        logger.info(f"Model loaded successfully. Output embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Encodes a single string or a list of text strings into 384-dimensional dense vectors.
        
        Args:
            texts: A string or list of text strings.
            batch_size: Batch size for batched forward passes.
            normalize_embeddings: If True, vectors are normalized to unit L2 length.
            show_progress_bar: Whether to display a tqdm progress bar.
            
        Returns:
            np.ndarray: Matrix of shape (N, 384) with float32 values.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


def precompute_and_save_embeddings(
    processed_dir: str,
    output_dir: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128
):
    """
    Precomputes and caches Sentence-BERT embeddings for train, validation, and test splits.
    Precomputing enables near-instant retraining and hyperparameter tuning of the MLP classifier
    without redundant Transformer forward passes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save label mappings for inference consistency
    mapping_path = os.path.join(output_dir, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({"label_to_int": LABEL_TO_INT, "int_to_label": INT_TO_LABEL}, f, indent=2)
    logger.info(f"Saved label mapping to: {mapping_path}")
    
    # Initialize encoder
    encoder = SentenceEncoder(model_name=model_name)
    
    splits = [
        ("train.csv", "train_embeddings.npy", "train_labels.npy"),
        ("validation.csv", "val_embeddings.npy", "val_labels.npy"),
        ("test.csv", "test_embeddings.npy", "test_labels.npy")
    ]
    
    for csv_file, emb_file, lbl_file in splits:
        csv_path = os.path.join(processed_dir, csv_file)
        emb_path = os.path.join(output_dir, emb_file)
        lbl_path = os.path.join(output_dir, lbl_file)
        
        logger.info(f"\n--- Processing {csv_file} ---")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing processed file: {csv_path}")
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} records from {csv_file}")
        
        # Extract text to encode
        texts = df["cleaned_text"].fillna("").tolist()
        
        # Compute embeddings
        logger.info(f"Encoding {len(texts):,} texts using {model_name} (batch_size={batch_size})...")
        embeddings = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        
        # Convert labels to integers
        labels = np.array([LABEL_TO_INT[y] for y in df["Y"].values], dtype=np.int64)
        
        # Save to disk
        np.save(emb_path, embeddings)
        np.save(lbl_path, labels)
        
        logger.info(f"Saved embeddings to {emb_path} (Shape: {embeddings.shape}, dtype: {embeddings.dtype})")
        logger.info(f"Saved labels to {lbl_path} (Shape: {labels.shape}, dtype: {labels.dtype})")
        
    logger.info("\nAll split embeddings successfully precomputed and cached in data/embeddings/!")


def load_cached_embeddings(
    output_dir: str,
    split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Utility to load cached embeddings and labels for a given split ('train', 'val', or 'test').
    """
    prefix = "val" if split in ["val", "validation"] else split
    emb_path = os.path.join(output_dir, f"{prefix}_embeddings.npy")
    lbl_path = os.path.join(output_dir, f"{prefix}_labels.npy")
    
    if not os.path.exists(emb_path) or not os.path.exists(lbl_path):
        raise FileNotFoundError(f"Cached embedding files not found at {emb_path} or {lbl_path}")
        
    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)
    return embeddings, labels


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc_dir = os.path.join(base_dir, "data", "processed")
    embed_dir = os.path.join(base_dir, "data", "embeddings")
    
    logger.info(f"Starting Phase 2 embedding precomputation...")
    logger.info(f"Source directory: {proc_dir}")
    logger.info(f"Target embeddings directory: {embed_dir}")
    
    precompute_and_save_embeddings(proc_dir, embed_dir, batch_size=128)
