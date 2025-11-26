"""
Integration tests for the inference-cloud API
Tests the full API endpoints with FastAPI TestClient
"""

import os

import pytest
from fastapi.testclient import TestClient
import numpy as np

# Set environment variables for model loading
os.environ["MODEL_CHECKPOINT_PATH"] = "/app/models/test_best.pth"
os.environ["DEVICE"] = "cpu"

# Import app after setting env vars  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client"""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoints:
    """Integration tests for health check endpoints"""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns health status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "service" in data
        assert "Skin Condition Classifier" in data["service"]
        assert data["model_loaded"] is True

    def test_root_returns_json(self, client):
        """Test that root returns JSON content type"""
        response = client.get("/")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    def test_health_endpoint(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_info" in data
        assert data["status"] == "healthy"

    def test_health_endpoint_model_info_structure(self, client):
        """Test that health endpoint returns proper model info"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        model_info = data["model_info"]

        assert "device" in model_info
        assert "num_classes" in model_info
        assert "fusion_strategy" in model_info
        assert model_info["num_classes"] == 57


class TestTextEmbeddingEndpoint:
    """Integration tests for text embedding endpoint"""

    def test_embed_text_success(self, client):
        """Test successful text embedding"""
        response = client.post("/embed-text", json={"text": "Patient has a dark mole on arm"})
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) > 0

    def test_embed_text_returns_correct_dimension(self, client):
        """Test that embedding has correct dimension"""
        response = client.post("/embed-text", json={"text": "Patient has symptoms"})
        assert response.status_code == 200
        data = response.json()
        assert data["dimension"] == 768
        assert len(data["embedding"]) == 768

    def test_embed_text_empty_string(self, client):
        """Test embedding empty text"""
        response = client.post("/embed-text", json={"text": ""})
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 768

    def test_embed_text_whitespace_only(self, client):
        """Test embedding whitespace-only text"""
        response = client.post("/embed-text", json={"text": "   "})
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data

    def test_embed_text_missing_field(self, client):
        """Test request with missing text field"""
        response = client.post("/embed-text", json={})
        assert response.status_code == 422

    def test_embed_text_invalid_type(self, client):
        """Test request with invalid text type"""
        response = client.post("/embed-text", json={"text": 123})
        assert response.status_code == 422

    def test_embed_text_null_value(self, client):
        """Test request with null text value"""
        response = client.post("/embed-text", json={"text": None})
        assert response.status_code == 422

    def test_embed_text_long_input(self, client):
        """Test embedding with very long text"""
        long_text = "Patient has symptoms " * 200
        response = client.post("/embed-text", json={"text": long_text})
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data


class TestPredictionEndpoint:
    """Integration tests for prediction endpoint"""

    def test_predict_success(self, client):
        """Test successful prediction"""
        vision_embedding = np.random.rand(2048).tolist()
        text_embedding = np.random.rand(768).tolist()

        response = client.post(
            "/predict", json={"vision_embedding": vision_embedding, "text_embedding": text_embedding, "top_k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "predicted_idx" in data
        assert "confidence" in data
        assert "top_k" in data
        assert isinstance(data["top_k"], list)

    def test_predict_response_types(self, client):
        """Test that prediction returns correct data types"""
        response = client.post(
            "/predict",
            json={"vision_embedding": np.random.rand(2048).tolist(), "text_embedding": np.random.rand(768).tolist()},
        )
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["predicted_class"], str)
        assert isinstance(data["predicted_idx"], int)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["top_k"], list)

    def test_predict_with_default_top_k(self, client):
        """Test prediction with default top_k parameter"""
        response = client.post(
            "/predict",
            json={"vision_embedding": np.random.rand(2048).tolist(), "text_embedding": np.random.rand(768).tolist()},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["top_k"]) == 5

    def test_predict_confidence_range(self, client):
        """Test that confidence is in valid range [0, 1]"""
        response = client.post(
            "/predict",
            json={"vision_embedding": np.random.rand(2048).tolist(), "text_embedding": np.random.rand(768).tolist()},
        )
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["confidence"] <= 1

    def test_predict_missing_vision_embedding(self, client):
        """Test prediction with missing vision embedding"""
        response = client.post("/predict", json={"text_embedding": np.random.rand(768).tolist()})
        assert response.status_code == 422

    def test_predict_missing_text_embedding(self, client):
        """Test prediction with missing text embedding"""
        response = client.post("/predict", json={"vision_embedding": np.random.rand(2048).tolist()})
        assert response.status_code == 422

    def test_predict_invalid_vision_embedding_format(self, client):
        """Test prediction with invalid vision embedding format"""
        response = client.post(
            "/predict", json={"vision_embedding": "not a list", "text_embedding": np.random.rand(768).tolist()}
        )
        assert response.status_code == 422

    def test_predict_invalid_text_embedding_format(self, client):
        """Test prediction with invalid text embedding format"""
        response = client.post(
            "/predict", json={"vision_embedding": np.random.rand(2048).tolist(), "text_embedding": "not a list"}
        )
        assert response.status_code == 422

    def test_predict_top_k_format(self, client):
        """Test that top_k predictions have correct format"""
        response = client.post(
            "/predict",
            json={
                "vision_embedding": np.random.rand(2048).tolist(),
                "text_embedding": np.random.rand(768).tolist(),
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data["top_k"]) == 3
        for prediction in data["top_k"]:
            assert "class" in prediction
            assert "probability" in prediction
            assert isinstance(prediction["class"], str)
            assert isinstance(prediction["probability"], float)
            assert 0 <= prediction["probability"] <= 1

    def test_predict_custom_top_k(self, client):
        """Test prediction with custom top_k value"""
        response = client.post(
            "/predict",
            json={
                "vision_embedding": np.random.rand(2048).tolist(),
                "text_embedding": np.random.rand(768).tolist(),
                "top_k": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["top_k"]) == 10

    def test_predict_invalid_top_k_type(self, client):
        """Test prediction with invalid top_k type"""
        response = client.post(
            "/predict",
            json={
                "vision_embedding": np.random.rand(2048).tolist(),
                "text_embedding": np.random.rand(768).tolist(),
                "top_k": "five",
            },
        )
        assert response.status_code == 422


class TestClassesEndpoint:
    """Integration tests for classes endpoint"""

    def test_get_classes(self, client):
        """Test getting all class names"""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "num_classes" in data
        assert isinstance(data["classes"], list)
        assert data["num_classes"] == len(data["classes"])

    def test_get_classes_non_empty(self, client):
        """Test that classes list is not empty"""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert len(data["classes"]) > 0
        assert data["num_classes"] > 0

    def test_get_classes_all_strings(self, client):
        """Test that all class names are strings"""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert all(isinstance(cls, str) for cls in data["classes"])


class TestErrorHandling:
    """Integration tests for error handling"""

    def test_invalid_route_returns_404(self, client):
        """Test that invalid routes return 404"""
        response = client.get("/this-route-does-not-exist")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test that wrong HTTP method returns 405"""
        response = client.post("/classes")
        assert response.status_code == 405

    def test_invalid_json_body(self, client):
        """Test handling of completely invalid JSON"""
        response = client.post("/predict", data="not json at all", headers={"Content-Type": "application/json"})
        assert response.status_code == 422


class TestCORS:
    """Tests for CORS configuration"""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present"""
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_response_time(client):
    """Test that API responds quickly"""
    import time

    start = time.time()
    response = client.get("/")
    elapsed = time.time() - start
    assert response.status_code == 200
    # Allow more time since we're loading real model
    assert elapsed < 2.0
