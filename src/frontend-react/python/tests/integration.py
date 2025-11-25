"""Integration tests for APIManager with real HTTP calls.

Tests are organized into 5 test classes:
- TestCloudAPIs: Cloud API status checks with REAL API calls (4 tests)
- TestPredictionFlow: End-to-end prediction workflow with MOCKED APIs (3 tests)
- TestChatFlow: Chat and followup functionality with MOCKED APIs (2 tests)
- TestDiseaseLifecycle: CRUD operations for diseases (2 tests)
- TestDataPersistence: Cross-instance data loading with MOCKED APIs (1 test)

Note: Only TestCloudAPIs makes real API calls. Other tests mock API responses for speed.
"""

import json
import pytest
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock

from api_manager import APIManager


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# ==================== Cloud API Tests (Status Only) ====================


class TestCloudAPIs:
    """Tests for cloud API endpoints (status code checks only)."""

    def test_cloud_text_embedding_api_returns_200(self, tmp_save_dir):
        """Test that text embedding endpoint returns 200 status."""
        case_id = "case_integration_001"
        manager = APIManager(case_id=case_id)

        text = "I have a red itchy rash on my arm"

        try:
            response = requests.post(manager.text_embed_url, json={"text": text}, timeout=60)
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API unavailable: {e}")

    @patch("api_manager.APIManager.get_vision_encoder")
    def test_cloud_prediction_api_returns_200(
        self, mock_get_encoder, tmp_save_dir, real_test_image, mock_vision_encoder
    ):
        """Test that prediction endpoint returns 200 status."""
        mock_get_encoder.return_value = mock_vision_encoder

        case_id = "case_integration_002"
        manager = APIManager(case_id=case_id)

        # Get real embeddings from vision encoder
        vision_embedding = manager._run_local_ml_model(real_test_image)

        # Get text embedding from text API
        try:
            text_response = requests.post(manager.text_embed_url, json={"text": "I have a red itchy rash"}, timeout=60)
            text_response.raise_for_status()
            text_embedding = text_response.json().get("embedding", [0.0] * 512)
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Text embedding API unavailable: {e}")

        # Test prediction API
        try:
            response = requests.post(
                manager.prediction_url,
                json={"vision_embedding": vision_embedding, "text_embedding": text_embedding, "top_k": 5},
                timeout=60,
            )
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Prediction API unavailable: {e}")

    def test_llm_explain_api_returns_200(self, tmp_save_dir, sample_predictions, sample_cv_analysis):
        """Test that LLM explain endpoint returns 200 status."""
        case_id = "case_integration_003"
        manager = APIManager(case_id=case_id)

        metadata = {"user_input": "I have a red itchy rash", "cv_analysis": sample_cv_analysis, "history": {}}

        try:
            response = requests.post(
                manager.llm_explain_url, json={"predictions": sample_predictions, "metadata": metadata}, timeout=300
            )
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API unavailable: {e}")

    def test_llm_followup_api_returns_200(self, tmp_save_dir):
        """Test that LLM followup endpoint returns 200 status."""
        case_id = "case_integration_004"
        manager = APIManager(case_id=case_id)

        initial_answer = "Based on your symptoms, it appears you may have eczema."
        question = "What treatments do you recommend?"
        conversation_history = []

        try:
            response = requests.post(
                manager.llm_followup_url,
                json={
                    "initial_answer": initial_answer,
                    "question": question,
                    "conversation_history": conversation_history,
                },
                timeout=60,
            )
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API unavailable: {e}")


# ==================== Full Prediction Flow ====================


class TestPredictionFlow:
    """Tests for end-to-end prediction workflow with mocked API responses."""

    @patch("requests.post")
    @patch("api_manager.APIManager.get_vision_encoder")
    def test_get_initial_prediction_end_to_end(
        self, mock_get_encoder, mock_requests_post, tmp_save_dir, real_test_image, mock_vision_encoder
    ):
        """Test full workflow with mocked API responses."""
        mock_get_encoder.return_value = mock_vision_encoder

        # Mock API responses
        def mock_post_response(url, **kwargs):
            response = MagicMock()
            response.status_code = 200

            # Text embedding response
            if "embed-text" in url:
                response.json.return_value = {"embedding": [0.1] * 512, "dimension": 512}
            # Prediction response
            elif "predict" in url:
                response.json.return_value = {
                    "top_k": [
                        {"class": "eczema", "probability": 0.78},
                        {"class": "contact_dermatitis", "probability": 0.15},
                        {"class": "psoriasis", "probability": 0.04},
                    ]
                }
            # LLM explain response
            elif "explain" in url:
                response.json.return_value = "Based on your symptoms and the analysis, it appears you may have eczema."
            # LLM followup response
            elif "followup" in url:
                response.json.return_value = {
                    "answer": "For eczema treatment, consider moisturizing regularly.",
                    "conversation_history": [],
                }

            return response

        mock_requests_post.side_effect = mock_post_response

        case_id = "case_integration_005"

        # Save body location first
        APIManager.save_body_location(case_id, {"coordinates": [45.5, 60.2], "nlp": "left elbow"})

        # Save demographics
        APIManager.save_demographics({"DOB": "1990-01-15", "Sex": "Female", "Race": "Asian", "Country": "USA"})

        manager = APIManager(case_id=case_id)

        result = manager.get_initial_prediction(
            image_path=real_test_image, text_description="I have a red itchy rash on my elbow"
        )

        # Verify result structure
        assert "llm_response" in result
        assert "predictions" in result
        assert "cv_analysis" in result
        assert "embedding" in result
        assert "enriched_disease" in result

        # Verify enriched disease has all fields
        disease = result["enriched_disease"]
        assert "id" in disease
        assert "name" in disease
        assert "bodyPart" in disease
        assert "confidenceLevel" in disease
        assert "llmResponse" in disease

    @patch("requests.post")
    @patch("api_manager.APIManager.get_vision_encoder")
    def test_get_initial_prediction_creates_all_files(
        self, mock_get_encoder, mock_requests_post, tmp_save_dir, real_test_image, mock_vision_encoder
    ):
        """Test that get_initial_prediction creates images/, case_history.json, conversation_history.json."""
        mock_get_encoder.return_value = mock_vision_encoder

        # Mock API responses (same as above)
        def mock_post_response(url, **kwargs):
            response = MagicMock()
            response.status_code = 200

            if "embed-text" in url:
                response.json.return_value = {"embedding": [0.1] * 512, "dimension": 512}
            elif "predict" in url:
                response.json.return_value = {"top_k": [{"class": "eczema", "probability": 0.78}]}
            elif "explain" in url:
                response.json.return_value = "Mock LLM response"

            return response

        mock_requests_post.side_effect = mock_post_response

        case_id = "case_integration_006"
        manager = APIManager(case_id=case_id)

        case_dir = tmp_save_dir / case_id

        manager.get_initial_prediction(image_path=real_test_image, text_description="Test description")

        # Check files created
        assert (case_dir / "images").exists()
        assert (case_dir / "case_history.json").exists()
        assert (case_dir / "conversation_history.json").exists()

        # Check at least one image saved
        images = list((case_dir / "images").glob("image_*.png"))
        assert len(images) >= 1

    @patch("requests.post")
    @patch("api_manager.APIManager.get_vision_encoder")
    def test_get_initial_prediction_returns_enriched_disease(
        self, mock_get_encoder, mock_requests_post, tmp_save_dir, real_test_image, mock_vision_encoder
    ):
        """Test that get_initial_prediction returns complete disease object with mocked predictions."""
        mock_get_encoder.return_value = mock_vision_encoder

        # Mock API responses
        def mock_post_response(url, **kwargs):
            response = MagicMock()
            response.status_code = 200

            if "embed-text" in url:
                response.json.return_value = {"embedding": [0.1] * 512, "dimension": 512}
            elif "predict" in url:
                response.json.return_value = {"top_k": [{"class": "psoriasis", "probability": 0.85}]}
            elif "explain" in url:
                response.json.return_value = "Mock LLM explanation"

            return response

        mock_requests_post.side_effect = mock_post_response

        case_id = "case_integration_007"

        # Setup location
        APIManager.save_body_location(case_id, {"coordinates": [50.0, 70.0], "nlp": "right arm"})

        manager = APIManager(case_id=case_id)

        result = manager.get_initial_prediction(image_path=real_test_image, text_description="Rash on my arm")

        disease = result["enriched_disease"]

        # Verify enrichment
        assert disease["bodyPart"] == "right arm"
        assert disease["mapPosition"]["leftPct"] == 50.0
        assert disease["mapPosition"]["topPct"] == 70.0
        assert disease["confidenceLevel"] > 0
        assert len(disease["timelineData"]) == 1
        assert len(disease["conversationHistory"]) == 1


# ==================== Chat Flow ====================


class TestChatFlow:
    """Tests for chat and followup functionality with mocked LLM."""

    @patch("requests.post")
    @patch("api_manager.APIManager.get_vision_encoder")
    def test_chat_message_with_mocked_llm(
        self, mock_get_encoder, mock_requests_post, tmp_save_dir, real_test_image, mock_vision_encoder
    ):
        """Test sending followup with mocked LLM API response."""
        mock_get_encoder.return_value = mock_vision_encoder

        # Mock API responses
        def mock_post_response(url, **kwargs):
            response = MagicMock()
            response.status_code = 200

            if "embed-text" in url:
                response.json.return_value = {"embedding": [0.1] * 512, "dimension": 512}
            elif "predict" in url:
                response.json.return_value = {"top_k": [{"class": "eczema", "probability": 0.78}]}
            elif "explain" in url:
                response.json.return_value = "Initial explanation about eczema."
            elif "followup" in url:
                response.json.return_value = {
                    "answer": "You should consult a dermatologist for proper treatment.",
                    "conversation_history": [],
                }

            return response

        mock_requests_post.side_effect = mock_post_response

        case_id = "case_integration_008"
        manager = APIManager(case_id=case_id)

        # First get initial prediction
        manager.get_initial_prediction(image_path=real_test_image, text_description="I have a rash")

        # Now send chat message
        chat_result = manager.chat_message("What should I do about it?")

        assert "answer" in chat_result
        assert "conversation_history" in chat_result
        assert isinstance(chat_result["answer"], str)
        assert len(chat_result["answer"]) > 0

        # Check conversation file updated
        conversation_file = tmp_save_dir / case_id / "conversation_history.json"
        with open(conversation_file, "r") as f:
            conversation = json.load(f)

        assert len(conversation) == 2  # Initial + followup

    def test_chat_message_fails_without_initial_prediction(self, tmp_save_dir):
        """Test that chat_message raises ValueError when no initial prediction exists."""
        case_id = "case_integration_009"
        manager = APIManager(case_id=case_id)

        with pytest.raises(ValueError) as exc_info:
            manager.chat_message("What should I do?")

        assert "No conversation found" in str(exc_info.value)


# ==================== Disease Lifecycle ====================


class TestDiseaseLifecycle:
    """Tests for CRUD operations on diseases and timeline entries."""

    def test_disease_create_load_update_cycle(self, tmp_save_dir):
        """Test full CRUD workflow for disease management."""
        # Create diseases
        diseases = [
            {"id": "101", "name": "Eczema", "image": "img1"},
            {"id": "102", "name": "Psoriasis", "image": "img2"},
        ]

        APIManager.save_diseases(diseases)

        # Create case histories for enrichment
        for disease in diseases:
            case_id = f"case_{disease['id']}"
            case_dir = tmp_save_dir / case_id
            case_dir.mkdir(parents=True)

            case_history = {
                "name": disease["name"],
                "dates": {
                    "2024-01-15": {
                        "image_path": "/path/to/image.png",
                        "text_summary": "Description",
                        "predictions": {disease["name"].lower(): 0.9},
                    }
                },
                "location": {"coordinates": [45, 60], "nlp": "arm"},
            }

            case_history_file = case_dir / "case_history.json"
            with open(case_history_file, "w") as f:
                json.dump(case_history, f)

        # Load diseases
        loaded = APIManager.load_diseases()
        assert len(loaded) == 2
        assert loaded[0]["name"] == "Eczema"
        assert loaded[1]["name"] == "Psoriasis"

        # Update disease name
        APIManager.update_disease_name("101", "Atopic Dermatitis")

        # Verify update
        loaded_again = APIManager.load_diseases()
        assert loaded_again[0]["name"] == "Atopic Dermatitis"

    def test_multiple_timeline_entries(self, tmp_save_dir):
        """Test that multiple dates can be added and retrieved."""
        case_id = "case_integration_010"
        case_dir = tmp_save_dir / case_id
        case_dir.mkdir(parents=True)

        # Create initial history
        APIManager.save_case_history(case_id, {"dates": {}, "location": {}})

        # Add multiple timeline entries
        dates = ["2024-01-15", "2024-01-20", "2024-01-25"]
        for i, date in enumerate(dates):
            APIManager.add_timeline_entry(
                case_id=case_id, image_path=f"/path/to/image_{i}.png", note=f"Note for day {i+1}", date=date
            )

        # Load and verify
        history = APIManager.load_case_history(case_id)

        assert len(history["dates"]) == 3
        for date in dates:
            assert date in history["dates"]


# ==================== Data Persistence ====================


class TestDataPersistence:
    """Tests for data persistence across APIManager instances."""

    @patch("requests.post")
    @patch("api_manager.APIManager.get_vision_encoder")
    def test_data_persists_across_instances(
        self, mock_get_encoder, mock_requests_post, tmp_save_dir, real_test_image, mock_vision_encoder
    ):
        """Test that new APIManager instance loads previously saved data correctly."""
        mock_get_encoder.return_value = mock_vision_encoder

        # Mock API responses
        def mock_post_response(url, **kwargs):
            response = MagicMock()
            response.status_code = 200

            if "embed-text" in url:
                response.json.return_value = {"embedding": [0.1] * 512, "dimension": 512}
            elif "predict" in url:
                response.json.return_value = {"top_k": [{"class": "dermatitis", "probability": 0.82}]}
            elif "explain" in url:
                response.json.return_value = "Explanation about dermatitis."

            return response

        mock_requests_post.side_effect = mock_post_response

        case_id = "case_integration_011"

        # Save demographics
        demographics = {"DOB": "1985-05-20", "Sex": "Male"}
        APIManager.save_demographics(demographics)

        # Save body location
        location = {"coordinates": [30.0, 40.0], "nlp": "chest"}
        APIManager.save_body_location(case_id, location)

        # First instance - create prediction
        manager1 = APIManager(case_id=case_id)

        manager1.get_initial_prediction(image_path=real_test_image, text_description="Rash on chest")

        # Close first instance (simulate app restart)
        del manager1

        # Second instance - should load all data
        manager2 = APIManager(case_id=case_id)

        # Verify demographics loaded
        assert manager2.demographics == demographics

        # Verify case history loaded
        assert len(manager2.case_history["dates"]) == 1
        assert manager2.case_history["location"] == location

        # Verify conversation loaded
        assert len(manager2.conversation_history) == 1
