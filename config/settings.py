"""
regiq Configuration
-------------------
All settings loaded from environment variables.
Copy .env.example to .env and fill in your values.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # ── LLM (vLLM / OpenAI-compatible) ───────────────────────────────────────
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cuda" on GPU

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8001"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "regiq_docs")
    chroma_persistent_path: Optional[str] = os.getenv("CHROMA_PERSISTENT_PATH") or None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── Prompts ───────────────────────────────────────────────────────────────
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8080"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton — import this everywhere
settings = Config()