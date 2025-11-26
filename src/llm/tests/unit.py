import pytest
from unittest.mock import Mock
from llm import LLM
from mock_data import (
    MOCK_PREDICTIONS,
    MOCK_METADATA,
    MOCK_ANSWER1,
    MOCK_QUESTION1,
    MOCK_QUESTION2,
    BASE_PROMPT,
    QUESTION_PROMPT,
)


class TestLLMModelLoading:
    """Test model loading and initialization."""

    def test_model_loading(self, mock_llm):
        """Test that LLM initializes with mocked components."""
        assert mock_llm is not None, "LLM model should be initialized"
        assert mock_llm.model is not None, "Model should be loaded"
        assert mock_llm.processor is not None, "Processor should be loaded"
        assert mock_llm.model_name == "medgemma-4b", "Model name should be medgemma-4b"

    def test_model_attributes(self, mock_llm):
        """Test that model has correct attributes set."""
        assert len(mock_llm.base_prompt) > 0, "Base prompt should not be empty"
        assert len(mock_llm.question_prompt) > 0, "Question prompt should not be empty"


class TestPromptBuilding:
    """Test prompt building functionality."""

    def test_build_prompt_with_all_components(self, mock_llm):
        """Test build_prompt includes all components."""
        prompt = mock_llm.build_prompt(MOCK_PREDICTIONS, MOCK_METADATA)

        # Check predictions are in prompt with percentages
        assert "eczema" in prompt.lower(), "Predictions should include 'eczema'"
        assert "78.0%" in prompt or "78%" in prompt, "Predictions should include percentage"

        # Check user input is in prompt
        assert "red, itchy patch" in prompt.lower(), "User input should be in prompt"
        assert "elbow" in prompt.lower(), "User input should be in prompt"

        # Check CV analysis is in prompt
        assert "area=" in prompt.lower() or "area" in prompt.lower(), "CV analysis should be in prompt"
        assert "8.4" in prompt, "CV analysis area value should be in prompt"
        assert "redness" in prompt.lower(), "CV analysis redness should be in prompt"
        assert "0.34" in prompt, "Redness index value should be in prompt"

        # Check history is in prompt
        assert "history" in prompt.lower(), "Historical data section should be in prompt"
        assert "2025-10-10" in prompt, "Historical date should be in prompt"
        assert "2025-11-01" in prompt, "Historical date should be in prompt"
        assert "12.6" in prompt, "Historical area value should be in prompt"

    def test_build_prompt_minimal(self, mock_llm):
        """Test build_prompt with minimal metadata."""
        metadata = {}
        prompt = mock_llm.build_prompt(MOCK_PREDICTIONS, metadata)

        # Should at least have predictions
        assert "eczema" in prompt.lower(), "Predictions should be in prompt"
        assert len(prompt) > 0, "Prompt should not be empty"


class TestGeneration:
    """Test text generation functionality."""

    def test_generate_returns_text(self, mock_llm):
        """Test that generate returns non-empty text."""
        prompt = "What are the symptoms of eczema?"
        response = mock_llm.generate(prompt, temperature=0.3)

        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"

    def test_generate_filters_thoughts(self, mock_llm, mock_processor):
        """Test that generate filters out 'thought' markers."""
        # Mock response with thought markers that should be filtered
        # Use a response without "thoughtful" to avoid false positives
        mock_processor.decode.return_value = (
            "<thought>internal reasoning</thought> This is the actual response about eczema treatment."
        )

        prompt = "Explain contact dermatitis briefly."
        response = mock_llm.generate(prompt, temperature=0.3)

        # Check that thought markers are removed
        assert "<thought>" not in response.lower(), "Response should not contain '<thought>' tags"
        assert "</thought>" not in response.lower(), "Response should not contain '</thought>' tags"
        # Check that the actual response content is present
        assert (
            "eczema" in response.lower() or "treatment" in response.lower()
        ), "Response should contain the actual content"


class TestExplain:
    """Test explain functionality."""

    def test_explain_generates_response(self, mock_llm):
        """Test that explain generates a substantive response."""
        response = mock_llm.explain(MOCK_PREDICTIONS, MOCK_METADATA, temperature=0.3)

        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should have substantive content"


class TestAskFollowup:
    """Test ask_followup functionality."""

    def test_ask_followup_returns_dict(self, mock_llm):
        """Test that ask_followup returns correct structure."""
        conversation_history = [MOCK_QUESTION1, MOCK_QUESTION2]

        result = mock_llm.ask_followup(
            initial_answer=MOCK_ANSWER1,
            question=MOCK_QUESTION1,
            conversation_history=conversation_history,
            temperature=0.3,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert "answer" in result, "Result should have 'answer' key"
        assert "conversation_history" in result, "Result should have 'conversation_history' key"

    def test_ask_followup_answer_content(self, mock_llm):
        """Test that ask_followup generates non-empty answer."""

        result = mock_llm.ask_followup(
            initial_answer=MOCK_ANSWER1, question=MOCK_QUESTION2, conversation_history=[], temperature=0.3
        )

        assert isinstance(result["answer"], str), "Answer should be a string"
        assert len(result["answer"]) > 0, "Answer should not be empty"

    def test_ask_followup_updates_history(self, mock_llm):
        """Test that ask_followup appends to conversation history."""
        conversation_history = [MOCK_QUESTION2]

        result = mock_llm.ask_followup(
            initial_answer=MOCK_ANSWER1,
            question=MOCK_QUESTION1,
            conversation_history=conversation_history.copy(),
            temperature=0.3,
        )

        assert len(result["conversation_history"]) == 2, "History should include previous question and new question"
        assert MOCK_QUESTION1 in result["conversation_history"], "New question should be in history"

    def test_ask_followup_empty_history(self, mock_llm):
        """Test ask_followup with no prior conversation history."""

        result = mock_llm.ask_followup(
            initial_answer=MOCK_ANSWER1,
            question=MOCK_QUESTION1,
            conversation_history=None,  # Test None case
            temperature=0.3,
        )

        assert len(result["conversation_history"]) == 1, "History should contain just the new question"
        assert result["conversation_history"][0] == MOCK_QUESTION1, "History should contain the question"
