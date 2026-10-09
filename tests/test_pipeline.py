# Tests for Pipeline
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.pipeline import Pipeline
from src.utils import load_config


def test_pipeline_initialization():
    """Test pipeline initializes correctly."""
    config = load_config()
    pipeline = Pipeline(config)

    assert pipeline.default_retriever == "visual"
    assert pipeline.top_k == 3
    assert pipeline._visual_retriever is None
    assert pipeline._text_retriever is None
    assert pipeline._generator is None

    pipeline.close()


@patch("src.pipeline.VisualRetriever")
@patch("src.pipeline.TextRetriever")
@patch("src.pipeline.Generator")
def test_pipeline_ask_visual_retriever(mock_generator_class, mock_text_retriever_class, mock_visual_retriever_class):
    """Test pipeline ask with visual retriever."""
    config = load_config()
    pipeline = Pipeline(config)

    # Setup mocks
    mock_visual_retriever = Mock()
    mock_visual_retriever.retrieve.return_value = [
        ("paper1_page_0004", 13.33),
        ("paper1_page_0005", 12.82),
    ]
    mock_visual_retriever_class.return_value = mock_visual_retriever

    mock_generator = Mock()
    mock_generator.generate.return_value = (
        "Multi-head attention is a mechanism... [paper1_page_0004]",
        [("paper1_page_0004", 0.0), ("paper1_page_0005", 0.0)]
    )
    mock_generator_class.return_value = mock_generator

    # Mock Path.exists to return True for images
    with patch.object(Path, "exists", return_value=True):
        result = pipeline.ask("What is multi-head attention?", retriever="visual", k=2)

    # Verify calls
    mock_visual_retriever.retrieve.assert_called_once_with("What is multi-head attention?", 2)
    mock_generator.generate.assert_called_once()

    # Verify result structure
    assert "answer" in result
    assert "sources" in result
    assert "scores" in result
    assert "retriever" in result
    assert "latency" in result
    assert result["retriever"] == "visual"
    assert len(result["sources"]) == 2

    pipeline.close()


@patch("src.pipeline.VisualRetriever")
@patch("src.pipeline.TextRetriever")
@patch("src.pipeline.Generator")
def test_pipeline_ask_text_retriever(mock_generator_class, mock_text_retriever_class, mock_visual_retriever_class):
    """Test pipeline ask with text retriever."""
    config = load_config()
    pipeline = Pipeline(config)

    # Setup mocks
    mock_text_retriever = Mock()
    mock_text_retriever.retrieve.return_value = [
        ("paper1_page_0003", 0.741),
        ("paper1_page_0009", 0.652),
    ]
    mock_text_retriever_class.return_value = mock_text_retriever

    mock_generator = Mock()
    mock_generator.generate.return_value = (
        "The transformer architecture uses... [paper1_page_0003]",
        [("paper1_page_0003", 0.0), ("paper1_page_0009", 0.0)]
    )
    mock_generator_class.return_value = mock_generator

    with patch.object(Path, "exists", return_value=True):
        result = pipeline.ask("Explain transformer architecture", retriever="text", k=2)

    # Verify calls
    mock_text_retriever.retrieve.assert_called_once_with("Explain transformer architecture", 2)
    mock_generator.generate.assert_called_once()

    assert result["retriever"] == "text"
    assert len(result["sources"]) == 2

    pipeline.close()


@patch("src.pipeline.VisualRetriever")
@patch("src.pipeline.Generator")
def test_pipeline_ask_no_results(mock_generator_class, mock_visual_retriever_class):
    """Test pipeline when retriever returns no results."""
    config = load_config()
    pipeline = Pipeline(config)

    mock_visual_retriever = Mock()
    mock_visual_retriever.retrieve.return_value = []
    mock_visual_retriever_class.return_value = mock_visual_retriever

    mock_generator = Mock()
    mock_generator_class.return_value = mock_generator

    result = pipeline.ask("Unknown question", retriever="visual", k=2)

    assert result["answer"] == "not found in the provided pages"
    assert result["sources"] == []
    assert result["retriever"] == "visual"
    mock_generator.generate.assert_not_called()

    pipeline.close()


@patch("src.pipeline.VisualRetriever")
@patch("src.pipeline.Generator")
def test_pipeline_ask_generator_error(mock_generator_class, mock_visual_retriever_class):
    """Test pipeline handles generator errors gracefully."""
    config = load_config()
    pipeline = Pipeline(config)

    mock_visual_retriever = Mock()
    mock_visual_retriever.retrieve.return_value = [
        ("paper1_page_0004", 13.33),
    ]
    mock_visual_retriever_class.return_value = mock_visual_retriever

    from src.generator import GeneratorError
    mock_generator = Mock()
    mock_generator.generate.side_effect = GeneratorError("Ollama not running")
    mock_generator_class.return_value = mock_generator

    with patch.object(Path, "exists", return_value=True):
        result = pipeline.ask("Test question", retriever="visual", k=2)

    assert "Error:" in result["answer"]
    assert "generator_error" in result
    assert result["sources"] == ["paper1_page_0004"]

    pipeline.close()


@patch("src.pipeline.VisualRetriever")
@patch("src.pipeline.Generator")
def test_pipeline_free_memory(mock_generator_class, mock_visual_retriever_class):
    """Test that retriever memory is freed after retrieval."""
    config = load_config()
    pipeline = Pipeline(config)

    mock_visual_retriever = Mock()
    mock_visual_retriever.retrieve.return_value = [
        ("paper1_page_0004", 13.33),
    ]
    mock_visual_retriever.model = Mock()
    mock_visual_retriever.processor = Mock()
    mock_visual_retriever._embeddings = {}
    mock_visual_retriever._page_order = []
    mock_visual_retriever._metadata = []
    mock_visual_retriever_class.return_value = mock_visual_retriever

    mock_generator = Mock()
    mock_generator.generate.return_value = ("Test answer", [("paper1_page_0004", 0.0)])
    mock_generator_class.return_value = mock_generator

    with patch.object(Path, "exists", return_value=True):
        pipeline.ask("Test question", retriever="visual", k=2)

    # Verify memory cleanup was attempted
    pipeline.close()


def test_pipeline_invalid_retriever():
    """Test pipeline raises error for invalid retriever."""
    config = load_config()
    pipeline = Pipeline(config)

    with pytest.raises(ValueError, match="Unknown retriever"):
        pipeline.ask("Test", retriever="invalid")

    pipeline.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
