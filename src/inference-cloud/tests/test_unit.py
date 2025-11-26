"""
Unit tests for InferenceClassifier
Tests the classifier methods with actual model checkpoint
"""

import pytest
import numpy as np
from inference_classifier import InferenceClassifier


@pytest.fixture(scope="module")
def classifier():
    """
    Load classifier once for all tests
    Uses the model checkpoint specified in Dockerfile
    """
    checkpoint_path = "/app/models/test_best.pth"
    device = "cpu"
    return InferenceClassifier(checkpoint_path=checkpoint_path, device=device)


class TestEmbedText:
    """Tests for the embed_text method"""

    def test_embed_text_returns_numpy_array(self, classifier):
        """Test that embed_text returns a numpy array"""
        result = classifier.embed_text("Patient has a dark mole on arm")
        assert isinstance(result, np.ndarray)

    def test_embed_text_correct_dimension(self, classifier):
        """Test that embeddings have correct dimensionality (768 for PubMedBERT)"""
        result = classifier.embed_text("Patient has a dark mole on arm")
        assert len(result.shape) == 1
        assert result.shape[0] == 768

    def test_embed_text_with_empty_string(self, classifier):
        """Test that empty text can be embedded"""
        result = classifier.embed_text("")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 768
        # Should not be all zeros (uses [MISSING] token)
        assert not np.allclose(result, 0)

    def test_embed_text_with_whitespace_only(self, classifier):
        """Test that whitespace-only text can be embedded"""
        result = classifier.embed_text("   ")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 768

    def test_embed_text_with_long_input(self, classifier):
        """Test embedding with long text"""
        long_text = "Patient has symptoms " * 100
        result = classifier.embed_text(long_text)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 768

    def test_embed_text_with_special_characters(self, classifier):
        """Test embedding with special characters"""
        text = "Patient has $$$ symptoms !@#$%"
        result = classifier.embed_text(text)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 768


class TestPredict:
    """Tests for the predict method"""

    def test_predict_returns_required_fields(self, classifier):
        """Test that predict returns all required fields"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict(vision_emb, text_emb)

        assert "predicted_class" in result
        assert "predicted_idx" in result
        assert "confidence" in result
        assert "top_k" in result

    def test_predict_returns_correct_types(self, classifier):
        """Test that predict returns correct types"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict(vision_emb, text_emb)

        assert isinstance(result["predicted_class"], str)
        assert isinstance(result["predicted_idx"], int)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["top_k"], list)

    def test_predict_confidence_in_valid_range(self, classifier):
        """Test that confidence is between 0 and 1"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict(vision_emb, text_emb)

        assert 0 <= result["confidence"] <= 1

    def test_predict_accepts_numpy_arrays(self, classifier):
        """Test that predict accepts numpy arrays"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict(vision_emb, text_emb)
        assert result is not None

    def test_predict_with_return_probs_false(self, classifier):
        """Test predict with return_probs=False"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict(vision_emb, text_emb, return_probs=False)

        assert "predicted_class" in result
        assert "confidence" in result
        assert "top_k" not in result


class TestPredictAsDict:
    """Tests for the predict_as_dict method"""

    def test_predict_as_dict_returns_dict(self, classifier):
        """Test that predict_as_dict returns a dictionary"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict_as_dict(vision_emb, text_emb)

        assert isinstance(result, dict)

    def test_predict_as_dict_format(self, classifier):
        """Test that dict format maps disease names to probabilities"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict_as_dict(vision_emb, text_emb, top_k=3)

        assert len(result) == 3
        for disease, prob in result.items():
            assert isinstance(disease, str)
            assert isinstance(prob, float)
            assert 0 <= prob <= 1


class TestGetClassNames:
    """Tests for the get_class_names method"""

    def test_get_class_names_returns_list(self, classifier):
        """Test that get_class_names returns a list"""
        result = classifier.get_class_names()
        assert isinstance(result, list)

    def test_get_class_names_non_empty(self, classifier):
        """Test that class names list is not empty"""
        result = classifier.get_class_names()
        assert len(result) > 0

    def test_get_class_names_all_strings(self, classifier):
        """Test that all class names are strings"""
        result = classifier.get_class_names()
        assert all(isinstance(name, str) for name in result)

    def test_get_class_names_expected_count(self, classifier):
        """Test that we have expected number of classes"""
        result = classifier.get_class_names()
        assert len(result) == 57


class TestGetModelInfo:
    """Tests for the get_model_info method"""

    def test_get_model_info_returns_dict(self, classifier):
        """Test that get_model_info returns a dictionary"""
        result = classifier.get_model_info()
        assert isinstance(result, dict)

    def test_get_model_info_required_fields(self, classifier):
        """Test that model info contains required fields"""
        result = classifier.get_model_info()

        assert "device" in result
        assert "num_classes" in result
        assert "classes" in result
        assert "text_encoder" in result
        assert "fusion_strategy" in result

    def test_get_model_info_num_classes_matches_list(self, classifier):
        """Test that num_classes matches class list length"""
        result = classifier.get_model_info()

        assert result["num_classes"] == len(result["classes"])
        assert result["num_classes"] == 57


class TestDeterminism:
    """Tests for deterministic behavior"""

    def test_embed_text_deterministic(self, classifier):
        """Test that same text produces same embedding"""
        text = "Patient has symptoms"

        emb1 = classifier.embed_text(text)
        emb2 = classifier.embed_text(text)

        assert np.allclose(emb1, emb2)

    def test_predict_deterministic(self, classifier):
        """Test that same inputs produce same predictions"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result1 = classifier.predict(vision_emb, text_emb)
        result2 = classifier.predict(vision_emb, text_emb)

        assert result1["predicted_class"] == result2["predicted_class"]
        assert result1["predicted_idx"] == result2["predicted_idx"]
        assert result1["confidence"] == result2["confidence"]


class TestTopK:
    """Tests for top_k parameter handling"""

    def test_predict_respects_top_k(self, classifier):
        """Test that predict returns correct number of predictions"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        for k in [1, 3, 5, 10]:
            result = classifier.predict(vision_emb, text_emb, top_k=k)
            assert len(result["top_k"]) == k

    def test_top_k_sorted_descending(self, classifier):
        """Test that top_k predictions are sorted by probability"""
        vision_emb = np.random.rand(2048).astype(np.float32)
        text_emb = np.random.rand(768).astype(np.float32)

        result = classifier.predict(vision_emb, text_emb, top_k=5)

        probabilities = [prob for _, prob in result["top_k"]]
        assert probabilities == sorted(probabilities, reverse=True)
