"""
DualQuery Modular Embedding Provider System.
Provides a unified interface for embedding generation across:
- EmbeddingGemma 2 (GPU via SentenceTransformers/PyTorch)
- FastEmbed (CPU via ONNX Runtime)
"""

from typing import Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the interface for all DualQuery embedding engines."""
    
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts (e.g. customer reviews)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for semantic search."""
        ...


class SentenceTransformerProvider:
    """
    GPU-accelerated embedding provider using local HuggingFace / SentenceTransformers models
    (Nomic-Embed-v1.5, EmbeddingGemma 2, or BGE-Base) with CUDA acceleration.
    """

    def __init__(self, model_name: str | None = None, dimension: int | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        from dualquery.config import settings

        self.model_name = model_name or settings.PRIMARY_EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformerProvider ({self.model_name}) on device: {self.device}")
        
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32}
        )

    def embed_documents(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings[:, :self.dimension].tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding[:self.dimension].tolist()


# Backward-compatibility alias
GemmaEmbeddingProvider = SentenceTransformerProvider


class FastEmbedProvider:
    """
    Lightweight, zero-GPU CPU fallback provider powered by ONNX Runtime.
    Default model: BAAI/bge-base-en-v1.5 (768 dimensions).
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", dimension: int = 768):
        from fastembed import TextEmbedding

        self.dimension = dimension
        logger.info(f"Loading FastEmbedProvider ({model_name}) on CPU ONNX Runtime")
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        # FastEmbed yields a generator of numpy arrays
        generator = self.model.embed(texts, batch_size=batch_size)
        return [vec[:self.dimension].tolist() for vec in generator]

    def embed_query(self, text: str) -> list[float]:
        # FastEmbed expects an iterable
        vec = next(self.model.embed([text]))
        return vec[:self.dimension].tolist()


def get_embedding_provider(preference: str = "auto") -> EmbeddingProvider:
    """
    Factory function to instantiate the optimal embedding provider based on
    available hardware and user preference.
    """
    if preference == "fastembed":
        return FastEmbedProvider()

    if preference == "gemma":
        return GemmaEmbeddingProvider()

    # 'auto' mode: check for CUDA hardware
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("NVIDIA CUDA detected. Activating GemmaEmbeddingProvider on GPU.")
            return GemmaEmbeddingProvider()
    except ImportError:
        pass

    logger.info("CUDA not available or torch missing. Defaulting to FastEmbed CPU ONNX.")
    return FastEmbedProvider()
