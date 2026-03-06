"""
Unit tests for ingestion.embedder
"""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.embedder import BGE_M3_MAX_CHARS, Embedder


class TestEmbedder:
    def _make_embedder_with_mock_model(self):
        """Return an Embedder whose internal model is stubbed."""
        embedder = Embedder.__new__(Embedder)
        embedder.model_name = "BAAI/bge-m3"
        embedder.device = "cpu"
        embedder.batch_size = 16
        embedder.normalize = True
        embedder._model = MagicMock()
        # model.encode returns a numpy-like list of float arrays
        embedder._model.encode = MagicMock(
            side_effect=lambda texts, **kw: [
                [0.1] * 1024 for _ in texts
            ]
        )
        return embedder

    def test_embed_returns_correct_count(self):
        emb = self._make_embedder_with_mock_model()
        texts = ["text one", "text two", "text three"]
        result = emb.embed(texts)
        assert len(result) == 3

    def test_embed_returns_correct_dimension(self):
        emb = self._make_embedder_with_mock_model()
        result = emb.embed(["hello world"])
        assert len(result[0]) == 1024

    def test_embed_one_returns_flat_vector(self):
        emb = self._make_embedder_with_mock_model()
        vec = emb.embed_one("single text")
        assert isinstance(vec, list)
        assert len(vec) == 1024

    def test_long_text_is_truncated_with_warning(self, caplog):
        emb = self._make_embedder_with_mock_model()
        long_text = "x" * (BGE_M3_MAX_CHARS + 100)
        import logging
        with caplog.at_level(logging.WARNING):
            result = emb.embed([long_text])
        assert len(result) == 1
        assert "truncated" in caplog.text.lower()

    def test_normal_text_not_truncated(self, caplog):
        emb = self._make_embedder_with_mock_model()
        import logging
        with caplog.at_level(logging.WARNING):
            emb.embed(["short text"])
        assert "truncated" not in caplog.text.lower()

    def test_empty_list_returns_empty(self):
        emb = self._make_embedder_with_mock_model()
        emb._model.encode = MagicMock(return_value=[])
        result = emb.embed([])
        assert result == []

    def test_dimension_property(self):
        emb = Embedder.__new__(Embedder)
        assert emb.dimension == 1024

    def test_model_loaded_lazily(self):
        """_model should be None until .model property is accessed."""
        emb = Embedder(model_name="BAAI/bge-m3", device="cpu")
        assert emb._model is None

    def test_model_load_raises_on_missing_dep(self):
        emb = Embedder.__new__(Embedder)
        emb.model_name = "BAAI/bge-m3"
        emb.device = "cpu"
        emb._model = None

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                emb._load_model()
