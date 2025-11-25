import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import numpy as np

from api_manager import APIManager

# ==================== Loading Tests ====================


class TestLoading:
    """Tests for loading operations (init, JSON loading, demographics, diseases)."""

    def test_init_creates_case_directory(self, tmp_save_dir):
        """Test that APIManager.__init__ creates case directory when it doesn't exist."""
        case_id = "case_test_001"
        manager = APIManager(case_id=case_id)

        case_dir = tmp_save_dir / case_id
        assert case_dir.exists()
        assert case_dir.is_dir()

    def test_init_loads_existing_case_history(self, tmp_save_dir):
        """Test that APIManager.__init__ loads existing case_history.json correctly."""
        case_id = "case_test_002"
        case_dir = tmp_save_dir / case_id
        case_dir.mkdir(parents=True)

        # Create existing case history
        case_history = {
            "dates": {
                "2024-01-15": {
                    "image_path": "/path/to/image.png",
                    "text_summary": "Red rash on arm",
                    "cv_analysis": {},
                    "predictions": {"eczema": 0.8},
                }
            },
            "location": {"coordinates": [45.5, 60.2], "nlp": "left arm"},
        }

        case_history_file = case_dir / "case_history.json"
        with open(case_history_file, "w") as f:
            json.dump(case_history, f)

        manager = APIManager(case_id=case_id)

        assert manager.case_history == case_history
        assert manager.case_history["dates"]["2024-01-15"]["predictions"]["eczema"] == 0.8

    def test_load_demographics_returns_data(self, tmp_save_dir, sample_demographics):
        """Test that APIManager.load_demographics returns saved demographics."""
        # Save demographics
        demographics_file = tmp_save_dir / "demographics.json"
        with open(demographics_file, "w") as f:
            json.dump(sample_demographics, f)

        loaded = APIManager.load_demographics()

        assert loaded == sample_demographics
        assert loaded["DOB"] == "1990-01-15"
        assert loaded["Sex"] == "Female"

    def test_load_diseases_enriches_from_case_history(self, tmp_save_dir):
        """Test that APIManager.load_diseases enriches disease data from case_history.json."""
        # Create diseases.json
        diseases = [{"id": "001", "name": "Test Disease", "image": "base64data"}]
        diseases_file = tmp_save_dir / "diseases.json"
        with open(diseases_file, "w") as f:
            json.dump(diseases, f)

        # Create case history for enrichment
        case_dir = tmp_save_dir / "case_001"
        case_dir.mkdir(parents=True)

        case_history = {
            "name": "Eczema",
            "dates": {
                "2024-01-15": {
                    "image_path": "/path/to/image.png",
                    "text_summary": "Red itchy rash",
                    "predictions": {"eczema": 0.85, "dermatitis": 0.10},
                }
            },
            "location": {"coordinates": [45.5, 60.2], "nlp": "left elbow"},
        }

        case_history_file = case_dir / "case_history.json"
        with open(case_history_file, "w") as f:
            json.dump(case_history, f)

        # Create conversation history
        conversation = [
            {
                "user": {"message": "I have a rash", "timestamp": "2024-01-15T10:00:00Z"},
                "llm": {"message": "Based on the analysis...", "timestamp": "2024-01-15T10:00:05Z"},
            }
        ]
        conversation_file = case_dir / "conversation_history.json"
        with open(conversation_file, "w") as f:
            json.dump(conversation, f)

        loaded = APIManager.load_diseases()

        assert len(loaded) == 1
        disease = loaded[0]
        assert disease["id"] == "001"
        assert disease["name"] == "Eczema"
        assert disease["bodyPart"] == "left elbow"
        assert disease["confidenceLevel"] == 85
        assert disease["llmResponse"] == "Based on the analysis..."


# ==================== Saving Tests ====================


class TestSaving:
    """Tests for saving operations (demographics, diseases, images, conversations)."""

    def test_save_demographics_creates_file(self, tmp_save_dir, sample_demographics):
        """Test that APIManager.save_demographics creates demographics.json with correct data."""
        APIManager.save_demographics(sample_demographics)

        demographics_file = tmp_save_dir / "demographics.json"
        assert demographics_file.exists()

        with open(demographics_file, "r") as f:
            loaded = json.load(f)

        assert loaded == sample_demographics

    def test_save_body_location_updates_existing_history(self, tmp_save_dir, sample_body_location):
        """Test that APIManager.save_body_location updates location while preserving dates."""
        case_id = "case_test_006"
        case_dir = tmp_save_dir / case_id
        case_dir.mkdir(parents=True)

        # Create existing history with dates
        existing_history = {
            "dates": {"2024-01-15": {"image_path": "/path/to/image.png", "text_summary": "test"}},
            "location": {"coordinates": [10, 20], "nlp": "old location"},
        }

        case_history_file = case_dir / "case_history.json"
        with open(case_history_file, "w") as f:
            json.dump(existing_history, f)

        APIManager.save_body_location(case_id, sample_body_location)

        with open(case_history_file, "r") as f:
            updated_history = json.load(f)

        assert updated_history["location"] == sample_body_location
        assert updated_history["dates"] == existing_history["dates"]  # Dates preserved

    def test_save_diseases_saves_minimal_fields(self, tmp_save_dir):
        """Test that APIManager.save_diseases saves only id, name, image (no duplication)."""
        diseases = [
            {
                "id": "001",
                "name": "Eczema",
                "image": "base64data",
                "bodyPart": "elbow",  # Extra field
                "confidenceLevel": 85,  # Extra field
                "description": "Some description",  # Extra field
            },
            {"id": "002", "name": "Psoriasis", "image": "base64data2"},
        ]

        APIManager.save_diseases(diseases)

        diseases_file = tmp_save_dir / "diseases.json"
        assert diseases_file.exists()

        with open(diseases_file, "r") as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        # Only minimal fields should be saved
        assert set(loaded[0].keys()) == {"id", "name", "image"}
        assert loaded[0]["id"] == "001"
        assert loaded[0]["name"] == "Eczema"

    def test_save_image_sequential_naming(self, tmp_save_dir, sample_image):
        """Test that APIManager._save_image names images sequentially (image_0001.png, image_0002.png, etc.)."""
        case_id = "case_test_008"
        manager = APIManager(case_id=case_id)

        img = Image.open(sample_image)

        # Save multiple images
        path1 = manager._save_image(img)
        path2 = manager._save_image(img)
        path3 = manager._save_image(img)

        assert "image_0001.png" in path1
        assert "image_0002.png" in path2
        assert "image_0003.png" in path3

    def test_save_conversation_entry_appends(self, tmp_save_dir):
        """Test that APIManager._save_conversation_entry appends to conversation list."""
        case_id = "case_test_009"
        manager = APIManager(case_id=case_id)

        # Add first entry
        manager._save_conversation_entry(
            user_message="Hello",
            llm_response="Hi there",
            user_timestamp="2024-01-15T10:00:00Z",
            llm_timestamp="2024-01-15T10:00:01Z",
        )

        assert len(manager.conversation_history) == 1
        assert manager.conversation_history[0]["user"]["message"] == "Hello"

        # Add second entry
        manager._save_conversation_entry(
            user_message="How are you?",
            llm_response="I'm doing well",
            user_timestamp="2024-01-15T10:01:00Z",
            llm_timestamp="2024-01-15T10:01:01Z",
        )

        assert len(manager.conversation_history) == 2
        assert manager.conversation_history[1]["user"]["message"] == "How are you?"

        # Verify file is saved
        conversation_file = tmp_save_dir / case_id / "conversation_history.json"
        assert conversation_file.exists()

    def test_save_history_entry_derives_name(self, tmp_save_dir, sample_predictions, sample_cv_analysis):
        """Test that APIManager._save_history_entry derives disease name from top prediction on first entry."""
        case_id = "case_test_010"
        manager = APIManager(case_id=case_id)

        # Add first entry (should derive name)
        manager._save_history_entry(
            date="2024-01-15",
            cv_analysis=sample_cv_analysis,
            predictions=sample_predictions,
            image_path="/path/to/image.png",
            text_summary="Red itchy rash",
        )

        assert "name" in manager.case_history
        assert manager.case_history["name"] == "Eczema"  # Top prediction formatted

    def test_update_disease_name_updates_history(self, tmp_save_dir):
        """Test that APIManager.update_disease_name updates name field in case_history.json."""
        case_id = "case_test_011"
        case_dir = tmp_save_dir / case_id
        case_dir.mkdir(parents=True)

        # Create initial history
        case_history = {"dates": {}, "location": {}, "name": "Old Name"}

        case_history_file = case_dir / "case_history.json"
        with open(case_history_file, "w") as f:
            json.dump(case_history, f)

        # Update name (handle both with and without case_ prefix)
        APIManager.update_disease_name("test_011", "New Disease Name")

        with open(case_history_file, "r") as f:
            updated = json.load(f)

        assert updated["name"] == "New Disease Name"

    def test_add_timeline_entry_adds_to_dates(self, tmp_save_dir):
        """Test that APIManager.add_timeline_entry adds new date entry to case_history.json."""
        case_id = "case_test_012"
        case_dir = tmp_save_dir / case_id
        case_dir.mkdir(parents=True)

        # Create initial history
        case_history = {
            "dates": {"2024-01-15": {"image_path": "/old/image.png", "text_summary": "old note"}},
            "location": {},
        }

        case_history_file = case_dir / "case_history.json"
        with open(case_history_file, "w") as f:
            json.dump(case_history, f)

        # Add new timeline entry
        APIManager.add_timeline_entry(
            case_id="test_012", image_path="/new/image.png", note="New note about condition", date="2024-01-20"
        )

        with open(case_history_file, "r") as f:
            updated = json.load(f)

        assert "2024-01-20" in updated["dates"]
        assert updated["dates"]["2024-01-20"]["image_path"] == "/new/image.png"
        assert updated["dates"]["2024-01-20"]["text_summary"] == "New note about condition"
        assert "2024-01-15" in updated["dates"]  # Old entry preserved


# ==================== Deletion Tests ====================


class TestDeletion:
    """Tests for deletion operations (reset all data)."""

    def test_reset_all_data(self, tmp_save_dir, sample_demographics):
        """Test that APIManager.reset_all_data removes demographics, diseases, and all case_* folders."""
        # Create demographics
        demographics_file = tmp_save_dir / "demographics.json"
        with open(demographics_file, "w") as f:
            json.dump(sample_demographics, f)

        # Create diseases
        diseases_file = tmp_save_dir / "diseases.json"
        with open(diseases_file, "w") as f:
            json.dump([{"id": "001", "name": "Test"}], f)

        # Create case directories
        (tmp_save_dir / "case_001").mkdir()
        (tmp_save_dir / "case_002").mkdir()
        (tmp_save_dir / "not_a_case_dir").mkdir()

        APIManager.reset_all_data()

        # Check all deleted
        assert not demographics_file.exists()
        assert not diseases_file.exists()
        assert not (tmp_save_dir / "case_001").exists()
        assert not (tmp_save_dir / "case_002").exists()
        # Non-case directories should remain
        assert (tmp_save_dir / "not_a_case_dir").exists()


# ==================== Text Processing Tests ====================


class TestTextProcessing:
    """Tests for text processing operations (demographic augmentation)."""

    def test_update_text_input_adds_age(self, tmp_save_dir, sample_demographics):
        """Test that APIManager.update_text_input calculates age from DOB and appends."""
        case_id = "case_test_013"

        # Save demographics
        demographics_file = tmp_save_dir / "demographics.json"
        with open(demographics_file, "w") as f:
            json.dump(sample_demographics, f)

        manager = APIManager(case_id=case_id)

        result = manager.update_text_input("I have a rash.")

        assert "I have a rash." in result
        assert "age is" in result.lower()
        assert "sex is female" in result.lower()

    def test_update_text_input_adds_location(self, tmp_save_dir, sample_body_location):
        """Test that APIManager.update_text_input appends body location to text."""
        case_id = "case_test_014"
        case_dir = tmp_save_dir / case_id
        case_dir.mkdir(parents=True)

        # Create case history with location
        case_history = {"dates": {}, "location": sample_body_location}

        case_history_file = case_dir / "case_history.json"
        with open(case_history_file, "w") as f:
            json.dump(case_history, f)

        manager = APIManager(case_id=case_id)

        result = manager.update_text_input("I have a rash.")

        assert "I have a rash." in result
        assert "body location" in result.lower()
        assert "left elbow" in result.lower()

    def test_update_text_input_no_demographics(self, tmp_save_dir):
        """Test that APIManager.update_text_input returns original text when no demographics exist."""
        case_id = "case_test_015"
        manager = APIManager(case_id=case_id)

        original = "I have a rash on my arm."
        result = manager.update_text_input(original)

        assert result == original


# ==================== Other Tests ====================


class TestOther:
    """Tests for vision encoder, ML model, and enriched disease building."""

    def test_get_vision_encoder_raises_when_checkpoint_missing(self, tmp_save_dir):
        """Test that APIManager.get_vision_encoder raises FileNotFoundError when checkpoint missing."""
        # The checkpoint path is hardcoded to look in python/inference_local/test_best.pth
        # Since we're in a test environment, it won't exist

        with pytest.raises(FileNotFoundError) as exc_info:
            APIManager.get_vision_encoder()

        assert "Model checkpoint not found" in str(exc_info.value)

    @patch("api_manager.APIManager.get_vision_encoder")
    def test_run_local_ml_model_returns_embedding(
        self, mock_get_encoder, tmp_save_dir, sample_image, mock_vision_encoder
    ):
        """Test that APIManager._run_local_ml_model returns embedding list (with mocked encoder)."""
        mock_get_encoder.return_value = mock_vision_encoder

        case_id = "case_test_016"
        manager = APIManager(case_id=case_id)

        embedding = manager._run_local_ml_model(sample_image)

        assert isinstance(embedding, list)
        assert len(embedding) == 2048  # ResNet101 embedding dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_build_enriched_disease_complete(self, tmp_save_dir, sample_predictions, sample_body_location):
        """Test that APIManager._build_enriched_disease populates all fields correctly."""
        case_id = "case_test_017"
        case_dir = tmp_save_dir / case_id
        case_dir.mkdir(parents=True)

        # Create complete case history
        case_history = {
            "name": "Eczema",
            "dates": {
                "2024-01-15": {
                    "image_path": str(tmp_save_dir / "test.png"),
                    "text_summary": "Red itchy rash on elbow",
                    "predictions": sample_predictions,
                    "cv_analysis": {},
                }
            },
            "location": sample_body_location,
        }

        case_history_file = case_dir / "case_history.json"
        with open(case_history_file, "w") as f:
            json.dump(case_history, f)

        # Create conversation
        conversation = [
            {
                "user": {"message": "I have a rash", "timestamp": "2024-01-15T10:00:00Z"},
                "llm": {"message": "Based on your symptoms...", "timestamp": "2024-01-15T10:00:05Z"},
            }
        ]
        conversation_file = case_dir / "conversation_history.json"
        with open(conversation_file, "w") as f:
            json.dump(conversation, f)

        manager = APIManager(case_id=case_id)

        enriched = manager._build_enriched_disease()

        assert enriched["id"] == "test_017"  # ID after removing 'case_' prefix
        assert enriched["name"] == "Eczema"
        assert enriched["bodyPart"] == "left elbow"
        assert enriched["confidenceLevel"] == 78
        assert enriched["llmResponse"] == "Based on your symptoms..."
        assert len(enriched["timelineData"]) == 1
        assert enriched["conversationHistory"] == conversation
