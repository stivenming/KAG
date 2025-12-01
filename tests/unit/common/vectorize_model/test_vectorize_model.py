# -*- coding: utf-8 -*-
import copy
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from kag.interface import VectorizeModelABC


@pytest.mark.skip(reason="Missing API key")
def test_openai_vectorize_model():
    conf = {
        "type": "openai",
        "model": "BAAI/bge-m3",
        "api_key": "",
        "base_url": "https://api.siliconflow.cn/v1/",
        "vector_dimensions": 1024,
    }
    vectorize_model = VectorizeModelABC.from_config(copy.deepcopy(conf))
    res1 = vectorize_model.vectorize("你好")
    res2 = asyncio.run(vectorize_model.avectorize("你好"))
    assert res1 is not None and res1 == res2


@pytest.mark.skip(reason="Missing model")
def test_ollama_vectorize_model():
    conf = {
        "type": "ollama",
        "model": "",
        "base_url": "http://127.0.0.1:11434/",
        "vector_dimensions": 1024,
    }
    vectorize_model = VectorizeModelABC.from_config(copy.deepcopy(conf))
    emb = vectorize_model.vectorize("你好")
    assert len(emb) == vectorize_model.get_vector_dimensions()


@pytest.mark.skip(reason="Missing model file")
def test_bge_vectorize_model():
    conf = {
        "type": "bge",
        "path": "~/.cache/vectorize_model/BAAI/bge-base-zh-v1.5",
        "url": "xxx",
        "vector_dimensions": 768,
    }

    vectorize_model = VectorizeModelABC.from_config(copy.deepcopy(conf))
    emb = vectorize_model.vectorize("你好")
    assert len(emb) == vectorize_model.get_vector_dimensions()

    vectorize_model2 = VectorizeModelABC.from_config(copy.deepcopy(conf))

    assert id(vectorize_model.model) == id(vectorize_model2.model)


@pytest.mark.skip(reason="Missing model file")
def test_bge_m3_vectorize_model():
    conf = {
        "type": "bge_m3",
        "path": "~/.cache/vectorize_model/BAAI/bge-m3",
        "url": "xxx",
        "vector_dimensions": 1024,
    }

    vectorize_model = VectorizeModelABC.from_config(copy.deepcopy(conf))
    emb = vectorize_model.vectorize("你好")
    assert len(emb) == vectorize_model.get_vector_dimensions()

    vectorize_model2 = VectorizeModelABC.from_config(copy.deepcopy(conf))

    assert id(vectorize_model.model) == id(vectorize_model2.model)


def test_mock_vectorize_model():
    conf = {
        "type": "mock",
        "vector_dimensions": 768,
    }
    vectorize_model = VectorizeModelABC.from_config(copy.deepcopy(conf))
    emb = vectorize_model.vectorize("你好")
    assert len(emb) == vectorize_model.get_vector_dimensions()
    embs = vectorize_model.vectorize(["你好", "再见"])
    assert len(embs) == 2
    for emb in embs:
        assert len(emb) == vectorize_model.get_vector_dimensions()


class TestOpenAIVectorizeModelBatching:
    """Tests for OpenAIVectorizeModel batching functionality."""

    @patch("kag.common.vectorize_model.openai_model.OpenAI")
    @patch("kag.common.vectorize_model.openai_model.AsyncOpenAI")
    def test_vectorize_batching_with_large_input(self, mock_async_openai, mock_openai):
        """Test that vectorize splits inputs larger than batch_size into chunks."""
        from kag.common.vectorize_model.openai_model import OpenAIVectorizeModel

        # Create mock embedding response
        def create_mock_response(texts):
            mock_items = []
            for _ in texts:
                mock_item = MagicMock()
                mock_item.embedding = [0.1, 0.2, 0.3]  # Mock embedding
                mock_items.append(mock_item)
            mock_response = MagicMock()
            mock_response.data = mock_items
            return mock_response

        # Setup mock client
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda **kwargs: create_mock_response(
            kwargs["input"]
        )
        mock_openai.return_value = mock_client

        # Create model with batch_size=3
        # Clear the singleton cache to ensure fresh instance
        VectorizeModelABC._instances.clear()
        model = OpenAIVectorizeModel(
            model="test-model",
            api_key="test-key",
            base_url="http://test.com",
            batch_size=3,
        )

        # Test with 7 texts (should result in 3 API calls: 3+3+1)
        texts = ["text1", "text2", "text3", "text4", "text5", "text6", "text7"]
        result = model.vectorize(texts)

        # Verify the result
        assert len(result) == 7
        for emb in result:
            assert emb == [0.1, 0.2, 0.3]

        # Verify that embeddings.create was called 3 times
        assert mock_client.embeddings.create.call_count == 3

        # Verify the batch sizes
        calls = mock_client.embeddings.create.call_args_list
        assert len(calls[0][1]["input"]) == 3  # First batch: 3 texts
        assert len(calls[1][1]["input"]) == 3  # Second batch: 3 texts
        assert len(calls[2][1]["input"]) == 1  # Third batch: 1 text

    @patch("kag.common.vectorize_model.openai_model.OpenAI")
    @patch("kag.common.vectorize_model.openai_model.AsyncOpenAI")
    def test_vectorize_batching_with_empty_strings(
        self, mock_async_openai, mock_openai
    ):
        """Test batching with empty strings in input."""
        from kag.common.vectorize_model.openai_model import OpenAIVectorizeModel

        # Create mock embedding response
        def create_mock_response(texts):
            mock_items = []
            for _ in texts:
                mock_item = MagicMock()
                mock_item.embedding = [0.1, 0.2, 0.3]
                mock_items.append(mock_item)
            mock_response = MagicMock()
            mock_response.data = mock_items
            return mock_response

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = lambda **kwargs: create_mock_response(
            kwargs["input"]
        )
        mock_openai.return_value = mock_client

        VectorizeModelABC._instances.clear()
        model = OpenAIVectorizeModel(
            model="test-model",
            api_key="test-key2",
            base_url="http://test2.com",
            batch_size=3,
        )

        # Test with 5 texts including empty strings
        texts = ["text1", "", "text3", "text4", ""]
        result = model.vectorize(texts)

        assert len(result) == 5
        assert result[0] == [0.1, 0.2, 0.3]  # Non-empty
        assert result[1] == []  # Empty string
        assert result[2] == [0.1, 0.2, 0.3]  # Non-empty
        assert result[3] == [0.1, 0.2, 0.3]  # Non-empty
        assert result[4] == []  # Empty string

        # Only 3 non-empty texts, batch_size=3, so 1 API call
        assert mock_client.embeddings.create.call_count == 1

    @patch("kag.common.vectorize_model.openai_model.OpenAI")
    @patch("kag.common.vectorize_model.openai_model.AsyncOpenAI")
    def test_vectorize_single_string_no_batching(self, mock_async_openai, mock_openai):
        """Test that single string input doesn't use batching logic."""
        from kag.common.vectorize_model.openai_model import OpenAIVectorizeModel

        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_item]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        VectorizeModelABC._instances.clear()
        model = OpenAIVectorizeModel(
            model="test-model",
            api_key="test-key3",
            base_url="http://test3.com",
            batch_size=3,
        )

        result = model.vectorize("single text")

        assert result == [0.1, 0.2, 0.3]
        assert mock_client.embeddings.create.call_count == 1

    @pytest.mark.asyncio
    @patch("kag.common.vectorize_model.openai_model.OpenAI")
    @patch("kag.common.vectorize_model.openai_model.AsyncOpenAI")
    async def test_avectorize_batching(self, mock_async_openai, mock_openai):
        """Test that avectorize splits inputs into batches."""
        from kag.common.vectorize_model.openai_model import OpenAIVectorizeModel

        # Create mock async embedding response
        async def create_mock_async_response(input, model):
            mock_items = []
            for _ in input:
                mock_item = MagicMock()
                mock_item.embedding = [0.1, 0.2, 0.3]
                mock_items.append(mock_item)
            mock_response = MagicMock()
            mock_response.data = mock_items
            return mock_response

        mock_async_client = MagicMock()
        mock_async_client.embeddings.create = AsyncMock(
            side_effect=create_mock_async_response
        )
        mock_async_openai.return_value = mock_async_client

        VectorizeModelABC._instances.clear()
        model = OpenAIVectorizeModel(
            model="test-model",
            api_key="test-key4",
            base_url="http://test4.com",
            batch_size=3,
        )

        # Test with 7 texts
        texts = ["text1", "text2", "text3", "text4", "text5", "text6", "text7"]
        result = await model.avectorize(texts)

        assert len(result) == 7
        for emb in result:
            assert emb == [0.1, 0.2, 0.3]

        # 7 texts, batch_size=3, so 3 API calls
        assert mock_async_client.embeddings.create.call_count == 3

    @patch("kag.common.vectorize_model.openai_model.OpenAI")
    @patch("kag.common.vectorize_model.openai_model.AsyncOpenAI")
    def test_default_batch_size(self, mock_async_openai, mock_openai):
        """Test that default batch_size is 10."""
        from kag.common.vectorize_model.openai_model import OpenAIVectorizeModel

        VectorizeModelABC._instances.clear()
        model = OpenAIVectorizeModel(
            model="test-model",
            api_key="test-key5",
            base_url="http://test5.com",
        )

        assert model.batch_size == 10
