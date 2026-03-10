"""
Evaluation Endpoints
---------------------
POST /v1/evaluate        — score a single (query, generated_answer, expected_answer) triple
POST /v1/evaluate/batch  — score up to 20 items, returns per-item + aggregate stats

Metrics:
  - ROUGE-1, ROUGE-2, ROUGE-L (F1)
  - Exact-match (case-insensitive, after stripping whitespace)
  - Answer length ratio (generated / expected)
"""

from __future__ import annotations

from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from generation.generator import RAGGenerator
from generation.models import GenerationRequest
from ..dependencies import get_generator

router = APIRouter(prefix="/v1", tags=["evaluation"])
logger = structlog.get_logger(__name__)

# ROUGE scorer — instantiated once at module level (no model download needed)
try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _SCORER = _rouge_scorer_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
except ImportError:
    _SCORER = None  # type: ignore[assignment]


# ── Pydantic models ───────────────────────────────────────────────────────────

class EvalItem(BaseModel):
    query: str
    generated_answer: str
    expected_answer: str
    source_filter: Optional[str] = None
    jurisdiction_filter: Optional[str] = None


class EvalScores(BaseModel):
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    exact_match: bool
    length_ratio: float     # len(generated) / len(expected) in words


class EvalResponse(BaseModel):
    query: str
    scores: EvalScores
    generated_answer: str
    expected_answer: str


class BatchEvalRequest(BaseModel):
    items: list[EvalItem] = Field(..., min_length=1, max_length=20)
    run_pipeline: bool = Field(
        False,
        description="If True, call the RAG pipeline for each item to get the generated answer "
                    "(ignores generated_answer field). Requires a running vLLM server.",
    )


class AggregateScores(BaseModel):
    mean_rouge1_f: float
    mean_rouge2_f: float
    mean_rougeL_f: float
    exact_match_rate: float
    mean_length_ratio: float
    n: int


class BatchEvalResponse(BaseModel):
    results: list[EvalResponse]
    aggregate: AggregateScores


# ── Scoring helper ────────────────────────────────────────────────────────────

def _score(generated: str, expected: str) -> EvalScores:
    if _SCORER is None:
        raise HTTPException(
            status_code=503,
            detail="rouge-score package is not installed. Run: pip install rouge-score",
        )

    scores = _SCORER.score(expected, generated)

    gen_words = len(generated.split())
    exp_words = len(expected.split()) or 1  # avoid /0

    return EvalScores(
        rouge1_f=round(scores["rouge1"].fmeasure, 4),
        rouge2_f=round(scores["rouge2"].fmeasure, 4),
        rougeL_f=round(scores["rougeL"].fmeasure, 4),
        exact_match=(generated.strip().lower() == expected.strip().lower()),
        length_ratio=round(gen_words / exp_words, 4),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvalResponse,
    summary="Score a single answer (ROUGE + exact match)",
)
async def evaluate(item: EvalItem) -> EvalResponse:
    scores = _score(item.generated_answer, item.expected_answer)
    return EvalResponse(
        query=item.query,
        scores=scores,
        generated_answer=item.generated_answer,
        expected_answer=item.expected_answer,
    )


@router.post(
    "/evaluate/batch",
    response_model=BatchEvalResponse,
    summary="Batch evaluation — up to 20 items with aggregate stats",
)
async def evaluate_batch(
    payload: BatchEvalRequest,
    generator: Annotated[RAGGenerator, Depends(get_generator)],
) -> BatchEvalResponse:
    results: list[EvalResponse] = []

    for item in payload.items:
        generated = item.generated_answer

        if payload.run_pipeline:
            try:
                req = GenerationRequest(
                    query=item.query,
                    source_filter=item.source_filter,
                    jurisdiction_filter=item.jurisdiction_filter,
                )
                resp = await generator.generate(req)
                generated = resp.answer
            except Exception as exc:
                logger.warning("eval_pipeline_error", query=item.query[:80], error=str(exc))
                generated = ""

        scores = _score(generated, item.expected_answer)
        results.append(EvalResponse(
            query=item.query,
            scores=scores,
            generated_answer=generated,
            expected_answer=item.expected_answer,
        ))

    n = len(results)
    agg = AggregateScores(
        mean_rouge1_f=round(sum(r.scores.rouge1_f for r in results) / n, 4),
        mean_rouge2_f=round(sum(r.scores.rouge2_f for r in results) / n, 4),
        mean_rougeL_f=round(sum(r.scores.rougeL_f for r in results) / n, 4),
        exact_match_rate=round(sum(1 for r in results if r.scores.exact_match) / n, 4),
        mean_length_ratio=round(sum(r.scores.length_ratio for r in results) / n, 4),
        n=n,
    )

    return BatchEvalResponse(results=results, aggregate=agg)
