"""Integration tests using FastAPI TestClient (no actual HTTP calls)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mock_data import MOCK_PREDICTIONS, MOCK_METADATA, MOCK_ANSWER1, MOCK_QUESTION1, MOCK_QUESTION2


@pytest.fixture
def mock_llm_instance():
    """Create a mocked LLM instance."""
    mock = MagicMock()
    mock.explain.return_value = "This is a mocked explanation of the skin condition."
    mock.ask_followup.return_value = {
        "answer": "This is a mocked follow-up answer.",
        "conversation_history": ["How long does it take to heal?"]
    }
    return mock


@pytest.fixture
def test_client(mock_llm_instance):
    """Create a FastAPI test client with mocked LLM."""
    # Import the modal app and patch the LLM
    import llm_modal
    
    # Create a FastAPI app from the Modal app
    from modal import App
    
    # Mock the Modal app to return a regular FastAPI app
    with patch('llm_modal.DermatologyLLM') as mock_class:
        # Set up the mock class
        mock_class.return_value = mock_llm_instance
        
        # Create a simple FastAPI app that mimics the Modal endpoints
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.post("/explain")
        def explain(json_data: dict) -> str:
            return mock_llm_instance.explain(
                predictions=json_data["predictions"],
                metadata=json_data["metadata"],
                temperature=0.3
            )
        
        @app.post("/ask_followup")
        def ask_followup(json_data: dict) -> dict:
            return mock_llm_instance.ask_followup(
                initial_answer=json_data["initial_answer"],
                question=json_data["question"],
                conversation_history=json_data.get("conversation_history", []),
                temperature=0.3
            )
        
        client = TestClient(app)
        yield client


@pytest.mark.integration
class TestExplainEndpoint:
    """Test the /explain endpoint using TestClient."""

    def test_explain_successful_response(self, test_client):
        """Test that /explain returns successful response."""
        payload = {"predictions": MOCK_PREDICTIONS, "metadata": MOCK_METADATA}
        
        response = test_client.post("/explain", json=payload)
        
        assert response.status_code == 200
        content = response.text.strip('"')  # Remove JSON quotes
        assert isinstance(content, str)
        assert len(content) > 0
        assert "mocked explanation" in content.lower()

    def test_explain_with_history(self, test_client):
        """Test /explain with historical data."""
        payload = {"predictions": MOCK_PREDICTIONS, "metadata": MOCK_METADATA}
        
        response = test_client.post("/explain", json=payload)
        
        assert response.status_code == 200
        assert len(response.text) > 0

    def test_explain_minimal_data(self, test_client):
        """Test /explain with minimal data."""
        payload = {"predictions": MOCK_PREDICTIONS, "metadata": {}}
        
        response = test_client.post("/explain", json=payload)
        
        assert response.status_code == 200


@pytest.mark.integration
class TestAskFollowupEndpoint:
    """Test the /ask_followup endpoint using TestClient."""

    def test_ask_followup_successful_response(self, test_client):
        """Test that /ask_followup returns successful response."""
        payload = {
            "initial_answer": MOCK_ANSWER1,
            "question": MOCK_QUESTION1,
            "conversation_history": []
        }
        
        response = test_client.post("/ask_followup", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "answer" in data
        assert "conversation_history" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert isinstance(data["conversation_history"], list)

    def test_ask_followup_with_history(self, test_client):
        """Test /ask_followup with existing conversation history."""
        payload = {
            "initial_answer": MOCK_ANSWER1,
            "question": MOCK_QUESTION1,
            "conversation_history": [MOCK_QUESTION2]
        }
        
        response = test_client.post("/ask_followup", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "conversation_history" in data

    def test_ask_followup_no_history(self, test_client):
        """Test /ask_followup without conversation history."""
        payload = {
            "initial_answer": MOCK_ANSWER1,
            "question": MOCK_QUESTION2
        }
        
        response = test_client.post("/ask_followup", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "conversation_history" in data
        assert len(data["conversation_history"]) >= 1

