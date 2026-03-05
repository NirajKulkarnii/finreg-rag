"""
EUR-Lex Loader
--------------
Loads EU financial and regulatory legislation from the MultiEURLEX dataset
hosted on HuggingFace (nlpaueb/multi_eurlex).

MultiEURLEX contains 65k+ EU legal documents across 23 languages with
EUROVOC concept labels. We filter specifically for:
  - Financial regulation topics (banking, securities, insurance, AML, payments)
  - English documents by default (multilingual option available)

Dataset card: https://huggingface.co/datasets/nlpaueb/multi_eurlex
Licence: CC BY 4.0 (free to use, must attribute)

Usage:
    loader = EURLexLoader()
    docs = loader.load(max_docs=200, financial_only=True)
"""

import logging
from typing import Generator, Optional

from ..models import RegulatoryDocument, DataSource

logger = logging.getLogger(__name__)


# EUROVOC concept IDs that map to financial / regulatory domains.
# Reference: https://eurovoc.europa.eu/
FINANCIAL_EUROVOC_IDS = {
    # Banking & finance
    "232",    # banking
    "2440",   # financial institution
    "4506",   # financial market
    "4508",   # capital market
    "4505",   # money market
    "1075",   # credit
    "4790",   # securities
    "4481",   # investment
    "4503",   # monetary policy
    "5859",   # payment system
    "4512",   # financial instrument
    # Insurance
    "5583",   # insurance
    "6030",   # life insurance
    # AML / financial crime
    "4516",   # money laundering
    "3030",   # financial fraud
    # Regulation & supervision
    "5547",   # financial supervision
    "4501",   # financial regulation
    "4791",   # stock exchange
    # Consumer finance
    "4489",   # consumer credit
    "5887",   # mortgage
}

# Human-readable category labels for known EUROVOC codes
CATEGORY_MAP = {
    "232": "banking",
    "2440": "banking",
    "4506": "financial_markets",
    "4508": "capital_markets",
    "4505": "money_markets",
    "1075": "credit",
    "4790": "securities",
    "4481": "investment",
    "4503": "monetary_policy",
    "5859": "payments",
    "4512": "financial_instruments",
    "5583": "insurance",
    "6030": "insurance",
    "4516": "aml",
    "3030": "financial_crime",
    "5547": "supervision",
    "4501": "regulation",
    "4791": "securities",
    "4489": "consumer_credit",
    "5887": "mortgages",
}


class EURLexLoader:
    """
    Loads EU regulatory documents from MultiEURLEX via HuggingFace datasets.

    Filtering behaviour:
      - financial_only=True  → only documents tagged with financial EUROVOC concepts
      - financial_only=False → all EU legislation (65k+ docs, useful for general RAG)
      - languages: list of ISO codes to load (default: ["en"])
        For multilingual builds, pass e.g. ["en", "fr", "de"]
    """

    HF_DATASET = "nlpaueb/multi_eurlex"

    def __init__(self):
        self._eurovoc_descriptors: Optional[dict] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        max_docs: int = 200,
        financial_only: bool = True,
        language: str = "en",
        split: str = "train",
    ) -> list[RegulatoryDocument]:
        """
        Load EUR-Lex documents.

        Args:
            max_docs:       Max documents to return.
            financial_only: If True, only load documents with financial EUROVOC labels.
            language:       ISO 639-1 code. MultiEURLEX supports 23 EU languages.
            split:          HuggingFace split — train/validation/test.

        Returns:
            List of RegulatoryDocument instances.
        """
        logger.info(
            f"EURLexLoader: loading up to {max_docs} docs "
            f"(language={language}, financial_only={financial_only})..."
        )
        docs = list(
            self._stream_documents(
                max_docs=max_docs,
                financial_only=financial_only,
                language=language,
                split=split,
            )
        )
        logger.info(f"EURLexLoader: loaded {len(docs)} documents.")
        return docs

    def stream(
        self,
        max_docs: int = 200,
        financial_only: bool = True,
        language: str = "en",
        split: str = "train",
    ) -> Generator[RegulatoryDocument, None, None]:
        """Streaming version — yields documents one at a time."""
        yield from self._stream_documents(
            max_docs=max_docs,
            financial_only=financial_only,
            language=language,
            split=split,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _stream_documents(
        self,
        max_docs: int,
        financial_only: bool,
        language: str,
        split: str,
    ) -> Generator[RegulatoryDocument, None, None]:
        """Loads dataset and yields filtered RegulatoryDocument instances."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library not installed. "
                "Run: pip install datasets"
            )

        logger.info(f"  Downloading nlpaueb/multi_eurlex ({language})...")
        dataset = load_dataset(
            self.HF_DATASET,
            language,
            split=split,
            trust_remote_code=True,
        )

        # Load EUROVOC label-to-string mapping
        label_feature = dataset.features["labels"].feature

        yielded = 0
        for sample in dataset:
            if yielded >= max_docs:
                break

            # Convert integer label IDs to EUROVOC string IDs
            try:
                eurovoc_ids = [
                    label_feature.int2str(label_id)
                    for label_id in sample["labels"]
                ]
            except Exception:
                eurovoc_ids = []

            # Filter to financial documents if requested
            if financial_only and not self._is_financial(eurovoc_ids):
                continue

            # Extract text — MultiEURLEX stores it as a plain string per language config
            text = sample.get("text", "")
            if isinstance(text, dict):
                # 'all_languages' config returns a dict keyed by lang code
                text = text.get(language, "")

            if not text or len(text.strip()) < 100:
                continue

            doc = self._build_document(
                celex_id=sample["celex_id"],
                text=text,
                eurovoc_ids=eurovoc_ids,
                language=language,
            )
            yield doc
            yielded += 1

            if yielded % 50 == 0:
                logger.debug(f"  EURLexLoader: {yielded} documents yielded so far...")

    def _is_financial(self, eurovoc_ids: list[str]) -> bool:
        """Returns True if any of the document's labels are in the financial set."""
        return bool(set(eurovoc_ids) & FINANCIAL_EUROVOC_IDS)

    def _build_document(
        self,
        celex_id: str,
        text: str,
        eurovoc_ids: list[str],
        language: str,
    ) -> RegulatoryDocument:
        """Constructs a RegulatoryDocument from a MultiEURLEX sample."""
        # Derive a readable category from the first matching financial label
        category = None
        for eid in eurovoc_ids:
            if eid in CATEGORY_MAP:
                category = CATEGORY_MAP[eid]
                break

        # Extract title from first line of text (EUR-Lex docs start with title)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        title = lines[0][:200] if lines else celex_id

        # Build EUR-Lex URL from CELEX ID
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex_id}"

        return RegulatoryDocument(
            doc_id=f"eurlex-{celex_id}",
            source=DataSource.EURLEX,
            title=title,
            text=text,
            url=url,
            jurisdiction="EU",
            language=language,
            category=category,
            metadata={
                "celex_id": celex_id,
                "eurovoc_ids": ",".join(eurovoc_ids),
            },
        )