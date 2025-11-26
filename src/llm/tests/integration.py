"""Integration tests for Modal API endpoints."""

import pytest
import os
import requests

from mock_data import MOCK_PREDICTIONS, MOCK_METADATA, MOCK_ANSWER1, MOCK_QUESTION1, MOCK_QUESTION2

API_EXPLAIN_URL = os.getenv("MODAL_API_EXPLAIN_URL")
API_ASK_FOLLOWUP_URL = os.getenv("MODAL_API_ASK_FOLLOWUP_URL")


@pytest.mark.integration
class TestExplainEndpoint:
    """Test the /explain endpoint."""

    def test_explain_successful_response(self):
        """Test that /explain returns successful response."""
        payload = {"predictions": MOCK_PREDICTIONS, "metadata": MOCK_METADATA}

        response = requests.post(API_EXPLAIN_URL, json=payload)

        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"

        # Response should be plain text (the explanation)
        content = response.text
        assert isinstance(content, str), "Response should be text"
        assert len(content) > 0, "Response should not be empty"

    def test_explain_with_history(self):
        """Test /explain with historical data."""
        payload = {"predictions": MOCK_PREDICTIONS, "metadata": MOCK_METADATA}

        response = requests.post(API_EXPLAIN_URL, json=payload)

        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        assert len(response.text) > 0, "Response should have content"

    def test_explain_minimal_data(self):
        """Test /explain with minimal data."""
        payload = {"predictions": MOCK_PREDICTIONS, "metadata": {}}

        response = requests.post(API_EXPLAIN_URL, json=payload)

        assert response.status_code == 200, f"Expected status 200 even with minimal data, got {response.status_code}"


@pytest.mark.integration
class TestAskFollowupEndpoint:
    """Test the /ask_followup endpoint."""

    def test_ask_followup_successful_response(self):
        """Test that /ask_followup returns successful response."""
        payload = {"initial_answer": MOCK_ANSWER1, "question": MOCK_QUESTION1, "conversation_history": []}

        response = requests.post(API_ASK_FOLLOWUP_URL, json=payload)

        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"

        # Response should be JSON
        data = response.json()
        assert isinstance(data, dict), "Response should be JSON object"
        assert "answer" in data, "Response should have 'answer' key"
        assert "conversation_history" in data, "Response should have 'conversation_history' key"

        # Check answer content
        assert isinstance(data["answer"], str), "Answer should be a string"
        assert len(data["answer"]) > 0, "Answer should not be empty"

        # Check conversation history
        assert isinstance(data["conversation_history"], list), "Conversation history should be a list"
        assert len(data["conversation_history"]) > 0, "Conversation history should contain the question"

    def test_ask_followup_with_history(self):
        """Test /ask_followup with existing conversation history."""
        payload = {
            "initial_answer": MOCK_ANSWER1,
            "question": MOCK_QUESTION1,
            "conversation_history": [MOCK_QUESTION1, MOCK_QUESTION2],
        }

        response = requests.post(API_ASK_FOLLOWUP_URL, json=payload)

        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"

        data = response.json()
        assert "answer" in data, "Response should have answer"
        assert "conversation_history" in data, "Response should have conversation history"

        # New question should be in history
        assert payload["question"] in data["conversation_history"], "New question should be in conversation history"

    def test_ask_followup_no_history(self):
        """Test /ask_followup without conversation history."""
        payload = {
            "initial_answer": MOCK_ANSWER1,
            "question": MOCK_QUESTION2,
            # Omitting conversation_history to test default empty list
        }

        response = requests.post(API_ASK_FOLLOWUP_URL, json=payload)

        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"

        data = response.json()
        assert "answer" in data, "Response should have answer"
        assert "conversation_history" in data, "Response should have conversation history"
        assert len(data["conversation_history"]) >= 1, "History should at least contain the new question"

    def test_ask_followup_multiple_questions(self):
        """Test multiple follow-up questions in sequence."""

        # First follow-up
        payload1 = {"initial_answer": MOCK_ANSWER1, "question": MOCK_QUESTION1, "conversation_history": []}

        response1 = requests.post(API_ASK_FOLLOWUP_URL, json=payload1)
        assert response1.status_code == 200
        data1 = response1.json()

        # Second follow-up using history from first
        payload2 = {
            "initial_answer": MOCK_ANSWER1,
            "question": MOCK_QUESTION2,
            "conversation_history": data1["conversation_history"],
        }

        response2 = requests.post(API_ASK_FOLLOWUP_URL, json=payload2)
        assert response2.status_code == 200
        data2 = response2.json()

        # Both questions should be in final history
        assert payload1["question"] in data2["conversation_history"], "First question should be in history"
        assert payload2["question"] in data2["conversation_history"], "Second question should be in history"
