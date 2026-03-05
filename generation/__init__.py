"""
Generation package — public API.

Quick start
-----------
    from retrieval import HybridRetriever
    from generation import RAGGenerator, GenerationRequest

    generator = RAGGenerator(retriever=hybrid_retriever)
    response  = await generator.generate(GenerationRequest(query="What is Consumer Duty?"))

    print(response.intent)
    print(response.answer)
    for src in response.sources:
        print(src.title, src.url)
"""

from .models import (
    GenerationRequest,
    GenerationResponse,
    Intent,
    Confidence,
    Source,
)
from .generator import RAGGenerator

__all__ = [
    "RAGGenerator",
    "GenerationRequest",
    "GenerationResponse",
    "Intent",
    "Confidence",
    "Source",
]
