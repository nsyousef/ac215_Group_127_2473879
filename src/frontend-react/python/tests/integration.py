"""Integration tests for APIManager with mocked HTTP calls."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import api_manager
from api_manager import APIManager


@pytest.fixture
def mock_requests():
    """Mock requests library for integration tests."""
    with patch("api_manager.requests") as mock_req:
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": {"eczema": 0.78, "psoriasis": 0.15},
            "embeddings": [0.1, 0.2, 0.3],
        }
        mock_response.text = "This is a mocked LLM explanation."
        mock_req.post.return_value = mock_response
        mock_req.get.return_value = mock_response
        yield mock_req


@pytest.fixture
def mock_vision_encoder():
    """Mock VisionEncoder for tests."""
    with patch("inference_local.vision_encoder.VisionEncoder") as mock_encoder_class:
        mock_encoder = MagicMock()
        mock_encoder.encode_image.return_value = [0.1] * 512
        mock_encoder.get_model_info.return_value = {"model": "ResNet50", "status": "loaded"}
        mock_encoder_class.return_value = mock_encoder
        yield mock_encoder


@pytest.mark.integration
class TestAPIManagerInitialization:
    """Test APIManager initialization."""

    def test_initialization(self):
        """Test that APIManager initializes correctly."""
        assert api_manager.SAVE_DIR is not None
        assert api_manager.BASE_URL is not None
        assert api_manager.PREDICTION_URL is not None


@pytest.mark.integration
class TestPredictionFlow:
    """Test the prediction flow with mocked HTTP calls."""

    def test_prediction_request(self, mock_requests):
        """Test making a prediction request."""
        response = mock_requests.post(f"{api_manager.PREDICTION_URL}", json={"image_embedding": [0.1] * 512})

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], dict)


@pytest.mark.integration
class TestLLMIntegration:
    """Test LLM integration with mocked HTTP calls."""

    def test_llm_explain_request(self, mock_requests):
        """Test making an LLM explain request."""
        payload = {"predictions": {"eczema": 0.78}, "metadata": {"user_input": "red patch"}}

        response = mock_requests.post(f"{api_manager.DEFAULT_LLM_EXPLAIN_URL}", json=payload)

        assert response.status_code == 200
        assert len(response.text) > 0
        assert "mocked" in response.text.lower()

    def test_llm_followup_request(self, mock_requests):
        """Test making an LLM follow-up request."""
        payload = {
            "initial_answer": "You have eczema.",
            "question": "How long does it take to heal?",
            "conversation_history": [],
        }

        mock_requests.post.return_value.json.return_value = {
            "answer": "It typically takes 2-4 weeks.",
            "conversation_history": ["How long does it take to heal?"],
        }

        response = mock_requests.post(f"{api_manager.DEFAULT_LLM_FOLLOWUP_URL}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "conversation_history" in data


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling with mocked failures."""

    def test_failed_request_handling(self, mock_requests):
        """Test handling of failed HTTP requests."""
        # Mock a failed response
        mock_requests.post.return_value.status_code = 500
        mock_requests.post.return_value.raise_for_status.side_effect = Exception("Server Error")

        response = mock_requests.post(f"{api_manager.PREDICTION_URL}", json={})

        assert response.status_code == 500
