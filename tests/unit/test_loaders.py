"""
Unit tests for data models and loader contracts.

Loader network calls are not made here — tests validate:
  - RegulatoryDocument schema and to_metadata()
  - DataSource enum values
  - Metadata completeness required by downstream components
"""

import pytest

from data.models import DataSource, RegulatoryDocument


class TestDataSource:
    def test_fca_value(self):
        assert DataSource.FCA.value == "fca"

    def test_eurlex_value(self):
        assert DataSource.EURLEX.value == "eurlex"

    def test_financebench_value(self):
        assert DataSource.FINANCEBENCH.value == "financebench"

    def test_from_string(self):
        assert DataSource("fca") == DataSource.FCA
        assert DataSource("eurlex") == DataSource.EURLEX

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DataSource("unknown_source")


class TestRegulatoryDocument:
    def _make(self, **overrides) -> RegulatoryDocument:
        defaults = dict(
            doc_id="test-001",
            source=DataSource.FCA,
            title="Test Policy Statement",
            text="Sample regulatory text.",
            url="https://example.com/doc",
            jurisdiction="UK",
            language="en",
            category="consumer_duty",
            date_published="2024-01-01",
        )
        defaults.update(overrides)
        return RegulatoryDocument(**defaults)

    def test_to_metadata_contains_required_keys(self):
        doc = self._make()
        meta = doc.to_metadata()
        required = {
            "doc_id", "source", "title", "url",
            "jurisdiction", "language", "category", "date_published",
        }
        assert required <= meta.keys()

    def test_to_metadata_source_is_string(self):
        """ChromaDB metadata must be flat strings — not enum objects."""
        doc = self._make(source=DataSource.FCA)
        meta = doc.to_metadata()
        assert meta["source"] == "fca"
        assert isinstance(meta["source"], str)

    def test_to_metadata_none_fields_become_empty_string(self):
        doc = self._make(url=None, jurisdiction=None, category=None,
                         date_published=None)
        meta = doc.to_metadata()
        assert meta["url"] == ""
        assert meta["jurisdiction"] == ""
        assert meta["category"] == ""
        assert meta["date_published"] == ""

    def test_optional_fields_default_to_none(self):
        doc = RegulatoryDocument(
            doc_id="x", source=DataSource.FCA, title="T", text="text"
        )
        assert doc.url is None
        assert doc.jurisdiction is None
        assert doc.category is None
        assert doc.date_published is None
        assert doc.eval_question is None
        assert doc.eval_answer is None

    def test_language_default_is_en(self):
        doc = RegulatoryDocument(
            doc_id="x", source=DataSource.FCA, title="T", text="t"
        )
        assert doc.language == "en"

    def test_metadata_dict_default_is_empty(self):
        doc = RegulatoryDocument(
            doc_id="x", source=DataSource.FCA, title="T", text="t"
        )
        assert doc.metadata == {}

    def test_extra_metadata_included_in_to_metadata(self):
        doc = self._make()
        doc.metadata = {"custom_field": "custom_value"}
        meta = doc.to_metadata()
        assert meta.get("custom_field") == "custom_value"

    def test_doc_id_preserved(self):
        doc = self._make(doc_id="fca-ps-23-3")
        assert doc.to_metadata()["doc_id"] == "fca-ps-23-3"

    def test_eval_fields_not_in_to_metadata(self):
        """eval_question/eval_answer are for evaluation only, not stored."""
        doc = self._make()
        doc.eval_question = "What is the duty?"
        doc.eval_answer = "To deliver good outcomes."
        meta = doc.to_metadata()
        assert "eval_question" not in meta
        assert "eval_answer" not in meta

    def test_title_preserved(self):
        doc = self._make(title="PS23/3: Consumer Duty")
        assert doc.to_metadata()["title"] == "PS23/3: Consumer Duty"

    def test_text_not_in_metadata(self):
        """Full text is too large for ChromaDB metadata."""
        doc = self._make(text="Very long document text...")
        meta = doc.to_metadata()
        assert "text" not in meta
