"""
FinanceBench Loader
-------------------
Loads the PatronusAI FinanceBench dataset for two purposes:

  1. KNOWLEDGE BASE: Source financial documents (SEC 10-Ks, 10-Qs, earnings
     reports from public companies) used to populate the RAG vector store.

  2. EVALUATION: 150 annotated QA pairs with human-verified answers used to
     benchmark retrieval quality and answer accuracy.

Dataset: https://huggingface.co/datasets/PatronusAI/financebench
Paper:   https://arxiv.org/abs/2311.11944
Licence: Apache 2.0

Usage:
    loader = FinanceBenchLoader()

    # Load source docs into vector store
    docs = loader.load_documents(max_docs=50)

    # Load QA pairs for evaluation
    eval_pairs = loader.load_eval_pairs()
    for pair in eval_pairs:
        print(pair.eval_question)
        print(pair.eval_answer)
"""

import logging
from typing import Generator

from ..models import RegulatoryDocument, DataSource

logger = logging.getLogger(__name__)

HF_DATASET = "PatronusAI/financebench"


class FinanceBenchLoader:
    """
    Loads FinanceBench documents and evaluation QA pairs.

    FinanceBench contains 150 annotated financial QA examples over real
    SEC filings and earnings reports from public US companies. It is
    used here as:
      - Source documents for the knowledge base (financial domain coverage)
      - Evaluation ground truth to measure RAG pipeline accuracy
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def load_documents(self, max_docs: int = 50) -> list[RegulatoryDocument]:
        """
        Load source financial documents suitable for RAG indexing.

        Each document is a segment of a real SEC filing, pre-chunked with
        its evidence string. Useful for adding US financial document coverage
        alongside UK/EU regulatory content.

        Args:
            max_docs: Max unique source documents to load.
        Returns:
            List of RegulatoryDocument instances (one per unique doc_name).
        """
        logger.info(f"FinanceBenchLoader: loading up to {max_docs} source documents...")
        dataset = self._load_hf_dataset()

        seen_docs: dict[str, RegulatoryDocument] = {}
        for sample in dataset:
            if len(seen_docs) >= max_docs:
                break
            doc_name = sample.get("doc_name", "")
            if doc_name in seen_docs:
                # Append additional evidence text to existing document
                seen_docs[doc_name].text += "\n\n" + sample.get("evidence", "")
            else:
                doc = self._build_source_document(sample)
                if doc:
                    seen_docs[doc_name] = doc

        docs = list(seen_docs.values())
        logger.info(f"FinanceBenchLoader: loaded {len(docs)} source documents.")
        return docs

    def load_eval_pairs(self) -> list[RegulatoryDocument]:
        """
        Load all 150 annotated QA evaluation pairs.

        Each returned document has:
          - text: the evidence passage (context the answer is grounded in)
          - eval_question: the question posed
          - eval_answer: the human-verified correct answer

        Used by evaluation/eval.py to score retrieval and generation quality.

        Returns:
            List of RegulatoryDocument instances with eval fields populated.
        """
        logger.info("FinanceBenchLoader: loading evaluation QA pairs...")
        dataset = self._load_hf_dataset()

        pairs = []
        for i, sample in enumerate(dataset):
            doc = self._build_eval_pair(sample, idx=i)
            if doc:
                pairs.append(doc)

        logger.info(f"FinanceBenchLoader: loaded {len(pairs)} evaluation pairs.")
        return pairs

    def stream_eval_pairs(self) -> Generator[RegulatoryDocument, None, None]:
        """Streaming version of load_eval_pairs."""
        dataset = self._load_hf_dataset()
        for i, sample in enumerate(dataset):
            doc = self._build_eval_pair(sample, idx=i)
            if doc:
                yield doc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_hf_dataset(self):
        """Downloads and returns the HuggingFace dataset split."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library not installed. "
                "Run: pip install datasets"
            )
        return load_dataset(HF_DATASET, split="train", trust_remote_code=True)

    def _build_source_document(self, sample: dict) -> RegulatoryDocument | None:
        """Builds a source document for knowledge base indexing."""
        doc_name = sample.get("doc_name", "")
        evidence = sample.get("evidence", "").strip()
        question = sample.get("question", "")

        if not evidence or len(evidence) < 50:
            return None

        # Derive a readable title from doc_name (e.g. "APPLE_2022_10K" → "Apple 2022 10-K")
        title = self._prettify_doc_name(doc_name)

        # Derive company ticker/name and document type from naming convention
        parts = doc_name.upper().split("_")
        company = parts[0] if parts else "Unknown"
        doc_type = parts[-1] if len(parts) >= 3 else "filing"
        year = parts[1] if len(parts) >= 2 and parts[1].isdigit() else ""

        return RegulatoryDocument(
            doc_id=f"financebench-{doc_name.lower().replace(' ', '-')}",
            source=DataSource.FINANCEBENCH,
            title=title,
            text=evidence,
            url=None,  # SEC EDGAR URL can be added via doc metadata if needed
            jurisdiction="US",
            language="en",
            category="financial_reporting",
            metadata={
                "doc_name": doc_name,
                "company": company,
                "year": year,
                "doc_type": doc_type,
                "sample_question": question[:200],  # Store for context
            },
        )

    def _build_eval_pair(self, sample: dict, idx: int) -> RegulatoryDocument | None:
        """Builds an evaluation document with question/answer fields populated."""
        evidence = sample.get("evidence", "").strip()
        question = sample.get("question", "").strip()
        answer = sample.get("answer", "").strip()
        doc_name = sample.get("doc_name", f"unknown_{idx}")

        if not question or not answer:
            return None

        return RegulatoryDocument(
            doc_id=f"financebench-eval-{idx}",
            source=DataSource.FINANCEBENCH,
            title=self._prettify_doc_name(doc_name),
            text=evidence,
            url=None,
            jurisdiction="US",
            language="en",
            category="financial_reporting",
            eval_question=question,
            eval_answer=answer,
            metadata={
                "doc_name": doc_name,
                "eval_idx": str(idx),
            },
        )

    @staticmethod
    def _prettify_doc_name(doc_name: str) -> str:
        """
        Converts doc_name like 'APPLE_2022_10K' to 'Apple 2022 10-K Annual Report'.
        """
        parts = doc_name.replace("_", " ").split()
        if not parts:
            return doc_name

        type_map = {
            "10K": "10-K Annual Report",
            "10Q": "10-Q Quarterly Report",
            "8K": "8-K Current Report",
            "DEF14A": "Proxy Statement",
        }

        prettified = []
        for part in parts:
            prettified.append(type_map.get(part.upper(), part.title()))
        return " ".join(prettified)