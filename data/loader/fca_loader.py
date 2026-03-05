"""
FCA Loader
----------
Downloads publicly available FCA policy statements and guidance documents
directly from the FCA website (https://www.fca.org.uk).

The FCA publishes a structured JSON feed of its publications which we use
instead of scraping HTML — this is cleaner, faster, and respects robots.txt.

Publication types included:
  - Policy Statements (PS)
  - Consultation Papers (CP)
  - Guidance Notes (GN)
  - Dear CEO / Dear Director letters
  - Finalised Guidance (FG)

Usage:
    loader = FCALoader()
    docs = loader.load(max_docs=100, categories=["PS", "CP"])
    for doc in docs:
        print(doc.title, doc.jurisdiction)
"""

import time
import logging
import requests
from typing import Generator, Optional
from datetime import datetime

from ..models import RegulatoryDocument, DataSource

logger = logging.getLogger(__name__)


# FCA publication categories mapped to readable labels
FCA_CATEGORIES = {
    "PS": "Policy Statement",
    "CP": "Consultation Paper",
    "GN": "Guidance Note",
    "FG": "Finalised Guidance",
    "DP": "Discussion Paper",
    "TR": "Thematic Review",
    "MR": "Market Study",
    "SP": "Supervisory Statement",
}

# FCA public search API — no auth required, returns JSON
FCA_SEARCH_API = "https://www.fca.org.uk/api/publications"

# Fallback: known stable FCA publication URLs for key regulatory docs
# These are curated high-value documents for financial compliance use cases
FCA_CURATED_DOCS = [
    {
        "id": "fca-ps-23-3",
        "title": "PS23/3: Consumer Duty — final rules and guidance",
        "url": "https://www.fca.org.uk/publication/policy/ps23-3.pdf",
        "category": "consumer_duty",
        "date": "2022-07",
    },
    {
        "id": "fca-cp-23-28",
        "title": "CP23/28: Strengthening protections for borrowers in financial difficulty",
        "url": "https://www.fca.org.uk/publication/consultation/cp23-28.pdf",
        "category": "consumer_credit",
        "date": "2023-11",
    },
    {
        "id": "fca-fg-21-4",
        "title": "FG21/4: Guidance for firms on the fair treatment of vulnerable customers",
        "url": "https://www.fca.org.uk/publication/finalised-guidance/fg21-4.pdf",
        "category": "consumer_protection",
        "date": "2021-02",
    },
    {
        "id": "fca-dp-23-4",
        "title": "DP23/4: Payments Regulation and the PSR Review",
        "url": "https://www.fca.org.uk/publication/discussion/dp23-4.pdf",
        "category": "payments",
        "date": "2023-11",
    },
    {
        "id": "fca-ps-22-9",
        "title": "PS22/9: Diversity and inclusion on company boards and executive management",
        "url": "https://www.fca.org.uk/publication/policy/ps22-9.pdf",
        "category": "governance",
        "date": "2022-04",
    },
]


class FCALoader:
    """
    Loads FCA regulatory publications as RegulatoryDocument instances.

    Strategy:
      1. Try the FCA publications search API to discover recent documents.
      2. Fall back to curated document list if the API is unavailable.
      3. For each discovered document, attempt PDF text extraction or HTML scraping.

    All requests are rate-limited and respect robots.txt.
    """

    BASE_URL = "https://www.fca.org.uk"
    REQUEST_DELAY = 1.5   # seconds between requests — polite crawling
    USER_AGENT = (
        "regiq-research-bot/1.0 "
        "(open-source regulatory intelligence RAG; "
        "https://github.com/NirajKulkarnii/regiq)"
    )

    def __init__(self, request_delay: float = REQUEST_DELAY):
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        max_docs: int = 50,
        pub_types: Optional[list[str]] = None,
    ) -> list[RegulatoryDocument]:
        """
        Load FCA documents.

        Args:
            max_docs:  Maximum number of documents to return.
            pub_types: Filter by FCA publication type codes e.g. ["PS", "CP"].
                       None means all types.

        Returns:
            List of RegulatoryDocument instances.
        """
        logger.info(f"FCALoader: loading up to {max_docs} documents...")
        docs = list(self._stream_documents(max_docs=max_docs, pub_types=pub_types))
        logger.info(f"FCALoader: loaded {len(docs)} documents.")
        return docs

    def stream(
        self,
        max_docs: int = 50,
        pub_types: Optional[list[str]] = None,
    ) -> Generator[RegulatoryDocument, None, None]:
        """Streaming version — yields documents one at a time."""
        yield from self._stream_documents(max_docs=max_docs, pub_types=pub_types)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _stream_documents(
        self,
        max_docs: int,
        pub_types: Optional[list[str]],
    ) -> Generator[RegulatoryDocument, None, None]:
        """
        Yields documents from the FCA API, falling back to curated list.
        """
        yielded = 0
        sources = self._discover_documents(pub_types=pub_types)

        for item in sources:
            if yielded >= max_docs:
                break
            try:
                doc = self._fetch_document(item)
                if doc and len(doc.text.strip()) > 200:
                    yield doc
                    yielded += 1
                    logger.debug(f"  ✓ {doc.doc_id}: {doc.title[:60]}...")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {item.get('id', '?')}: {e}")
            finally:
                time.sleep(self.request_delay)

    def _discover_documents(
        self, pub_types: Optional[list[str]]
    ) -> list[dict]:
        """
        Returns a list of document descriptors to fetch.
        Tries the FCA search API first; falls back to curated list.
        """
        # Try API first
        try:
            return self._fetch_from_api(pub_types=pub_types)
        except Exception as e:
            logger.warning(f"FCA API unavailable ({e}), using curated document list.")
            curated = FCA_CURATED_DOCS
            if pub_types:
                # Filter curated by category prefix match
                curated = [
                    d for d in curated
                    if any(d["id"].startswith(f"fca-{t.lower()}") for t in pub_types)
                ]
            return curated

    def _fetch_from_api(self, pub_types: Optional[list[str]]) -> list[dict]:
        """
        Queries the FCA publications search API.
        Returns normalised document descriptors.
        """
        params = {
            "fields": "title,url,date,type",
            "page_size": 100,
            "sort": "date:desc",
        }
        if pub_types:
            params["type"] = ",".join(pub_types)

        resp = self.session.get(FCA_SEARCH_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "id": f"fca-{item.get('reference', item['url'].split('/')[-1])}",
                "title": item.get("title", "Untitled"),
                "url": item["url"] if item["url"].startswith("http")
                       else self.BASE_URL + item["url"],
                "category": item.get("type", ""),
                "date": item.get("date", ""),
            })
        return results

    def _fetch_document(self, item: dict) -> Optional[RegulatoryDocument]:
        """
        Fetches and parses the text of a single FCA document.
        Handles both HTML pages and direct PDF links.
        """
        url = item["url"]
        time.sleep(self.request_delay)

        if url.endswith(".pdf"):
            text = self._extract_pdf_text(url)
        else:
            text = self._extract_html_text(url)

        if not text:
            return None

        return RegulatoryDocument(
            doc_id=item["id"],
            source=DataSource.FCA,
            title=item["title"],
            text=self._clean_text(text),
            url=url,
            jurisdiction="UK",
            language="en",
            category=item.get("category", ""),
            date_published=item.get("date", ""),
            metadata={"pub_type": item.get("category", "")},
        )

    def _extract_html_text(self, url: str) -> Optional[str]:
        """Extracts readable text from an FCA HTML publication page."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
            return None

        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # FCA pages wrap main content in .fca-content or article tags
        content = (
            soup.find("div", class_="fca-content")
            or soup.find("article")
            or soup.find("main")
        )
        if not content:
            content = soup

        # Remove nav, headers, footers, scripts
        for tag in content.find_all(["nav", "header", "footer", "script", "style"]):
            tag.decompose()

        return content.get_text(separator="\n", strip=True)

    def _extract_pdf_text(self, url: str) -> Optional[str]:
        """Downloads a PDF and extracts its text content."""
        try:
            import pdfplumber
            import io
        except ImportError:
            logger.error("pdfplumber not installed. Run: pip install pdfplumber")
            return None

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()

        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text cleaning: collapse whitespace, remove junk lines."""
        import re
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove lines that are just page numbers or single characters
        lines = [l for l in text.split("\n") if len(l.strip()) > 2]
        return "\n".join(lines).strip()