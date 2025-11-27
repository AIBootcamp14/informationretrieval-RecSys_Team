"""
Embedding module with GPU optimization and domain-specific models
"""
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np
from loguru import logger
import config.config as cfg


class Embedder:
    """Enhanced embedder with GPU support and batch optimization"""

    def __init__(
        self,
        model_name: str = cfg.EMBEDDING_MODEL,
        device: str = cfg.EMBEDDING_DEVICE,
        batch_size: int = cfg.EMBEDDING_BATCH_SIZE
    ):
        """
        Initialize embedder

        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            batch_size: Batch size for encoding (optimized for RTX 3090)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load model with explicit device
        self.model = SentenceTransformer(model_name, device=self.device)

        # Enable GPU optimizations
        if self.device == 'cuda':
            # Note: BGE-M3 may not support FP16 well, so we skip .half()
            # self.model.half()  # Disabled for BGE-M3 compatibility
            logger.info("Using GPU with FP32 precision for BGE-M3")

        logger.info(f"Embedding dimension: {self.get_dimension()}")

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Override default batch size
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device
        )

        return embeddings

    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Encode texts in batches with progress tracking

        Args:
            texts: List of texts
            show_progress: Show progress bar

        Returns:
            List of embedding arrays
        """
        logger.info(f"Encoding {len(texts)} texts in batches of {self.batch_size}")

        embeddings = self.encode(
            texts,
            batch_size=self.batch_size,
            show_progress=show_progress
        )

        return embeddings


class EnsembleEmbedder:
    """Ensemble multiple embedding models for better coverage"""

    def __init__(self, model_names: List[str], weights: List[float] = None):
        """
        Initialize ensemble embedder

        Args:
            model_names: List of model names
            weights: Weights for each model (default: equal weights)
        """
        self.embedders = [Embedder(name) for name in model_names]
        self.weights = weights or [1.0 / len(model_names)] * len(model_names)

        logger.info(f"Initialized ensemble with {len(self.embedders)} models")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode texts using ensemble

        Args:
            texts: Single text or list of texts

        Returns:
            Weighted average of embeddings
        """
        all_embeddings = []

        for embedder, weight in zip(self.embedders, self.weights):
            emb = embedder.encode(texts)
            all_embeddings.append(emb * weight)

        # Weighted average
        ensemble_emb = np.sum(all_embeddings, axis=0)

        # Normalize
        ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb, axis=-1, keepdims=True)

        return ensemble_emb


# Singleton instance for efficient reuse
_embedder_instance = None

def get_embedder() -> Embedder:
    """Get singleton embedder instance"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
