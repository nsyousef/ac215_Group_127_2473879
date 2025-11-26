"""
System tests for cloud inference API
Tests the entire deployed system with real HTTP requests
Requires the service to be running on Cloud Run
"""

import pytest
import requests
import time
import numpy as np


# Base URL for the deployed API
API_BASE_URL = "https://inference-cloud-469023639150.us-east4.run.app"


def is_api_running():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except Exception:  # Fixed: catch Exception instead of bare except
        return False


@pytest.mark.skipif(not is_api_running(), reason="API not running on Cloud Run")
class TestSystemEndToEnd:
    """System tests for complete workflows"""

    def test_health_check_system(self):
        """System test: Basic health check"""
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_detailed_health_check(self):
        """System test: Detailed health endpoint with model info"""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "model_info" in data
        assert data["status"] == "healthy"

        # Verify model info structure
        model_info = data["model_info"]
        assert "device" in model_info
        assert "num_classes" in model_info
        assert "fusion_strategy" in model_info

    def test_complete_prediction_workflow(self):
        """System test: Complete end-to-end prediction workflow"""
        # Step 1: Generate text embedding from patient description
        text_response = requests.post(
            f"{API_BASE_URL}/embed-text", json={"text": "Patient has a dark asymmetric mole with irregular borders"}
        )
        assert text_response.status_code == 200
        text_data = text_response.json()
        text_embedding = text_data["embedding"]

        # Verify text embedding structure
        assert isinstance(text_embedding, list)
        assert len(text_embedding) == 768  # PubMedBERT dimension

        # Step 2: Generate fake vision embedding (in production, from vision model)
        vision_embedding = np.random.rand(2048).tolist()

        # Step 3: Make prediction with both embeddings
        predict_response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"vision_embedding": vision_embedding, "text_embedding": text_embedding, "top_k": 5},
        )
        assert predict_response.status_code == 200
        predict_data = predict_response.json()

        # Verify prediction structure
        assert "predicted_class" in predict_data
        assert "predicted_idx" in predict_data
        assert "confidence" in predict_data
        assert "top_k" in predict_data
        assert len(predict_data["top_k"]) == 5

        # Verify prediction values are valid
        assert isinstance(predict_data["predicted_class"], str)
        assert 0 <= predict_data["confidence"] <= 1
        assert all(0 <= p["probability"] <= 1 for p in predict_data["top_k"])

    def test_multiple_sequential_predictions(self):
        """System test: Multiple predictions in sequence"""
        vision_emb = np.random.rand(2048).tolist()

        # Make 5 predictions with different text descriptions
        descriptions = [
            "melanoma symptoms",
            "nevus characteristics",
            "basal cell carcinoma",
            "squamous cell carcinoma",
            "benign mole",
        ]

        for i, desc in enumerate(descriptions):
            # Get text embedding
            text_response = requests.post(f"{API_BASE_URL}/embed-text", json={"text": desc})
            assert text_response.status_code == 200

            # Make prediction
            predict_response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"vision_embedding": vision_emb, "text_embedding": text_response.json()["embedding"]},
            )
            assert predict_response.status_code == 200

            # Verify each prediction is valid
            data = predict_response.json()
            assert "predicted_class" in data
            assert 0 <= data["confidence"] <= 1

    def test_get_all_classes(self):
        """System test: Retrieve all disease classes"""
        response = requests.get(f"{API_BASE_URL}/classes")
        assert response.status_code == 200
        data = response.json()

        assert "classes" in data
        assert "num_classes" in data
        assert len(data["classes"]) > 0
        assert data["num_classes"] == len(data["classes"])
        assert all(isinstance(cls, str) for cls in data["classes"])

    def test_text_embedding_variations(self):
        """System test: Text embeddings with different inputs"""
        test_cases = [
            "Patient has melanoma",
            "",  # Empty string
            "Very long description " * 50,  # Long text
            "Special chars !@#$%",
        ]

        for text in test_cases:
            response = requests.post(f"{API_BASE_URL}/embed-text", json={"text": text})
            assert response.status_code == 200
            data = response.json()
            assert len(data["embedding"]) == 768

    def test_prediction_with_different_top_k(self):
        """System test: Predictions with different top_k values"""
        vision_emb = np.random.rand(2048).tolist()
        text_emb = np.random.rand(768).tolist()

        for top_k in [1, 3, 5, 10]:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"vision_embedding": vision_emb, "text_embedding": text_emb, "top_k": top_k},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["top_k"]) == top_k


@pytest.mark.skipif(not is_api_running(), reason="API not running")
class TestSystemPerformance:
    """System tests for performance characteristics"""

    def test_response_time_health(self):
        """Test that health check responds quickly"""
        start = time.time()
        response = requests.get(f"{API_BASE_URL}/")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 2.0  # Should respond within 2 seconds

    def test_response_time_embedding(self):
        """Test that text embedding responds in reasonable time"""
        start = time.time()
        response = requests.post(f"{API_BASE_URL}/embed-text", json={"text": "Patient has melanoma symptoms on arm"})
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 10.0  # Should complete within 10 seconds

    def test_response_time_prediction(self):
        """Test that prediction responds in reasonable time"""
        vision_emb = np.random.rand(2048).tolist()
        text_emb = np.random.rand(768).tolist()

        start = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict", json={"vision_embedding": vision_emb, "text_embedding": text_emb}
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 5.0  # Prediction should be reasonably fast

    def test_concurrent_requests(self):
        """Test that API can handle multiple concurrent requests"""
        import concurrent.futures

        def make_request():
            response = requests.get(f"{API_BASE_URL}/")
            return response.status_code == 200

        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(results)


@pytest.mark.skipif(not is_api_running(), reason="API not running")
class TestSystemErrorHandling:
    """System tests for error handling"""

    def test_invalid_endpoint(self):
        """Test 404 for invalid endpoint"""
        response = requests.get(f"{API_BASE_URL}/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_request_format(self):
        """Test 422 for invalid request format"""
        response = requests.post(f"{API_BASE_URL}/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test 422 for missing required fields"""
        # Missing text_embedding
        response = requests.post(f"{API_BASE_URL}/predict", json={"vision_embedding": np.random.rand(2048).tolist()})
        assert response.status_code == 422

    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        response = requests.post(
            f"{API_BASE_URL}/embed-text", data="not valid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_invalid_embedding_dimensions(self):
        """Test error handling for wrong embedding dimensions"""
        # Wrong vision embedding dimension
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "vision_embedding": np.random.rand(512).tolist(),  # Wrong: should be 2048
                "text_embedding": np.random.rand(768).tolist(),
            },
        )
        # May return 422 or 500 depending on validation
        assert response.status_code in [422, 500]

    def test_empty_embeddings(self):
        """Test error handling for empty embeddings"""
        response = requests.post(f"{API_BASE_URL}/predict", json={"vision_embedding": [], "text_embedding": []})
        assert response.status_code in [422, 500]

    def test_method_not_allowed(self):
        """Test 405 for wrong HTTP method"""
        response = requests.post(f"{API_BASE_URL}/classes")  # Should be GET
        assert response.status_code == 405
