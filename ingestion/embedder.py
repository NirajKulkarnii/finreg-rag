"""
BGE-M3 Embedder
---------------
Wraps sentence-transformers to produce dense embeddings using BAAI/bge-m3.

BGE-M3 outputs 1024-dim vectors and handles texts up to 8192 tokens,
making it well suited for long regulatory documents.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# BGE-M3 max sequence length
BGE_M3_MAX_TOKENS = 8192
# Safe character limit to avoid truncation (≈ tokens × 3.5 chars/token)
BGE_M3_MAX_CHARS = 8192 * 3


class Embedder:
    """
    Produces dense embeddings using BAAI/bge-m3 via sentence-transformers.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(["text one", "text two"])
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings. Returns a list of float vectors.

        Texts longer than BGE-M3's context window are truncated with a warning.
        """
        truncated = []
        for t in texts:
            if len(t) > BGE_M3_MAX_CHARS:
                logger.warning(
                    f"Text truncated from {len(t)} to {BGE_M3_MAX_CHARS} chars "
                    f"(BGE-M3 max context)."
                )
                truncated.append(t[:BGE_M3_MAX_CHARS])
            else:
                truncated.append(t)

        vectors = self.model.encode(
            truncated,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding dimension (1024 for BGE-M3)."""
        return 1024

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info(f"Loading embedding model {self.model_name} on {self.device}...")
        model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded.")
        return model
