"""Shared pytest fixtures for LLM tests."""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock

# Mock torch before importing llm module to avoid ModuleNotFoundError
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Add parent directory to path to import llm and prompts modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LLM  # noqa: E402


@pytest.fixture
def mock_model():
    """Create a mock model for testing without loading actual weights."""
    mock = MagicMock()
    mock.parameters.return_value = iter([Mock(device=Mock(type="cpu"))])
    return mock


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing."""
    mock = MagicMock()
    mock.apply_chat_template.return_value = "formatted_prompt"

    # The processor is called and then .to() is called on the result
    # Create a mock that supports .to() method
    mock_processor_result = MagicMock()
    mock_processor_result.to.return_value = mock_processor_result  # .to() returns self
    mock_processor_result.__getitem__.return_value = Mock(shape=(1, 10))  # For inputs['input_ids']
    mock.return_value = mock_processor_result

    mock.pad_token_id = 0
    mock.eos_token_id = 1
    mock.decode.return_value = "This is a mocked response from the model."
    mock.encode.return_value = [123, 456]  # Mock tokens for "thought"
    return mock


@pytest.fixture
def mock_llm(mocker, mock_model, mock_processor):
    """Create an LLM instance with mocked model and processor."""
    from prompts import BASE_PROMPT, QUESTION_PROMPT

    # Mock the model loading
    mocker.patch("llm.AutoProcessor.from_pretrained", return_value=mock_processor)
    mocker.patch("llm.AutoModelForImageTextToText.from_pretrained", return_value=mock_model)

    # Mock torch.cuda
    mocker.patch("llm.torch.cuda.is_available", return_value=False)

    # Create LLM instance (will use mocked components)
    llm_instance = LLM(
        model_name="medgemma-4b", max_new_tokens=700, base_prompt=BASE_PROMPT, question_prompt=QUESTION_PROMPT
    )

    # Mock the generate method to return predictable output
    # The generate method does: outputs[0][input_length:]
    # Create a simple nested list structure: outputs[0] returns a list that can be sliced
    # outputs is a list containing one element (a list of token IDs)
    mock_token_ids = [1, 2, 3, 4, 5]  # This will be sliced: [input_length:]
    mock_output = [mock_token_ids]  # outputs[0] returns mock_token_ids which can be sliced
    llm_instance.model.generate.return_value = mock_output

    # Mock the processor.decode to return a proper string response
    mock_processor.decode.return_value = "This is a mocked response from the model."

    # Mock the generate method at the LLM instance level to return a string
    def mock_generate_impl(prompt, temperature=0.7):
        """Mock implementation that returns a string."""
        return "This is a mocked diagnosis response based on the provided information. The diagnosis is eczema."

    llm_instance.generate = mock_generate_impl

    return llm_instance
