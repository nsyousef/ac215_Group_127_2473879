from ml_workflow.inference.vision_encoder import VisionEncoder
from ml_workflow.inference.inference_classifier import InferenceClassifier
from PIL import Image
import numpy as np

def test_vision_encoder():
    print("="*60)
    print("Vision Encoder Testing")
    print("="*60)

    # Test 1: Can it initialize from checkpoint?
    print("\n[Test 1] Initialization from checkpoint...")
    try:
        encoder = VisionEncoder(
            checkpoint_path="ml_workflow/inference/test_best.pth",
            device="cpu"  # Use CPU for quick testing
        )
        print("✓ Initialization successful")
        model_info = encoder.get_model_info()
        print(f"  Device: {model_info['device']}")
        print(f"  Model: {model_info['model_name']}")
        print(f"  Embedding dim: {encoder.get_embedding_dim()}")
        print(f"  Image size: {model_info['img_size']}")
        print(f"  Pooling: {model_info['pooling_type']}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 2: Can it encode an image?
    print("\n[Test 2] Encoding a simple test image...")
    try:
        test_image = Image.new('RGB', (224, 224), color='red')
        embedding = encoder.encode(test_image)
        print(f"✓ Encoding successful")
        print(f"  Shape: {embedding.shape}")
        print(f"  Dtype: {embedding.dtype}")
        print(f"  Sample values: {embedding[:5]}")
        print(f"  Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")
        
        # Check shape matches expected embedding dimension
        expected_dim = encoder.get_embedding_dim()
        assert embedding.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {embedding.shape}"
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 3: Does it produce consistent embeddings?
    print("\n[Test 3] Consistency check (same image should give same embedding)...")
    try:
        embedding1 = encoder.encode(test_image)
        embedding2 = encoder.encode(test_image)
        diff = np.abs(embedding1 - embedding2).max()
        print(f"✓ Max difference: {diff}")
        assert diff < 1e-6, f"Embeddings not consistent! Max diff: {diff}"
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 4: Different images should give different embeddings
    print("\n[Test 4] Different images produce different embeddings...")
    try:
        blue_image = Image.new('RGB', (224, 224), color='blue')
        emb_red = encoder.encode(test_image)
        emb_blue = encoder.encode(blue_image)
        
        cosine_sim = np.dot(emb_red, emb_blue) / (np.linalg.norm(emb_red) * np.linalg.norm(emb_blue))
        print(f"✓ Cosine similarity: {cosine_sim:.4f}")
        assert cosine_sim < 0.99, f"Embeddings too similar! Similarity: {cosine_sim}"
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "="*60)
    print("✅ ALL VISION ENCODER TESTS PASSED!")
    print("="*60)
    
    return encoder, test_image


def test_inference_classifier(vision_encoder, test_image):
    print("\n" + "="*60)
    print("Inference Classifier Testing")
    print("="*60)

    # Test 1: Initialize classifier from checkpoint
    print("\n[Test 1] Initialization from checkpoint...")
    try:
        classifier = InferenceClassifier(
            checkpoint_path="ml_workflow/inference/test_best.pth",
            device="cpu"
        )
        print("✓ Initialization successful")
        model_info = classifier.get_model_info()
        print(f"  Device: {model_info['device']}")
        print(f"  Number of classes: {model_info['num_classes']}")
        print(f"  Sample classes: {model_info['classes'][:5]}")
        print(f"  Fusion strategy: {model_info['fusion_strategy']}")
        print(f"  Text encoder: {model_info['text_encoder']}")
        print(f"  Text pooling: {model_info['text_pooling']}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 2: Text embedding
    print("\n[Test 2] Embedding text description...")
    try:
        test_text = "Patient has red, itchy rash on arms with scaling"
        text_embedding = classifier.embed_text(test_text)
        print(f"✓ Text embedding successful")
        print(f"  Shape: {text_embedding.shape}")
        print(f"  Dtype: {text_embedding.dtype}")
        print(f"  Sample values: {text_embedding[:5]}")
        print(f"  Min: {text_embedding.min():.4f}, Max: {text_embedding.max():.4f}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 3: Text embedding consistency
    print("\n[Test 3] Text embedding consistency check...")
    try:
        text_emb1 = classifier.embed_text(test_text)
        text_emb2 = classifier.embed_text(test_text)
        diff = np.abs(text_emb1 - text_emb2).max()
        print(f"✓ Max difference: {diff}")
        assert diff < 1e-6, f"Text embeddings not consistent! Max diff: {diff}"
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 4: Different texts produce different embeddings
    print("\n[Test 4] Different texts produce different embeddings...")
    try:
        text1 = "Red itchy rash"
        text2 = "Blue nevus lesion"
        emb1 = classifier.embed_text(text1)
        emb2 = classifier.embed_text(text2)
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"✓ Cosine similarity: {cosine_sim:.4f}")
        assert cosine_sim < 0.95, f"Text embeddings too similar! Similarity: {cosine_sim}"
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 5: End-to-end prediction
    print("\n[Test 5] End-to-end prediction...")
    try:
        # Get vision embedding from test image
        vision_embedding = vision_encoder.encode(test_image)
        
        # Get text embedding
        text_description = "Red inflamed skin with visible rash"
        text_embedding = classifier.embed_text(text_description)
        
        # Make prediction
        result = classifier.predict(
            vision_embedding=vision_embedding,
            text_embedding=text_embedding,
            return_probs=True,
            top_k=5
        )
        
        print(f"✓ Prediction successful")
        print(f"  Predicted class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Top 5 predictions:")
        for i, (class_name, prob) in enumerate(result['top_k'], 1):
            print(f"    {i}. {class_name}: {prob:.4f}")
        
        # Validate result structure
        assert 'predicted_class' in result
        assert 'predicted_idx' in result
        assert 'confidence' in result
        assert 'top_k' in result
        assert len(result['top_k']) == 5
        assert 0 <= result['confidence'] <= 1
        
        # Check probabilities sum to ~1
        total_prob = sum(prob for _, prob in result['top_k'])
        print(f"  Sum of top-5 probs: {total_prob:.4f}")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 6: predict_as_dict format
    print("\n[Test 6] Testing predict_as_dict format...")
    try:
        result_dict = classifier.predict_as_dict(
            vision_embedding=vision_embedding,
            text_embedding=text_embedding,
            top_k=3
        )
        
        print(f"✓ predict_as_dict successful")
        print(f"  Result type: {type(result_dict)}")
        print(f"  Number of predictions: {len(result_dict)}")
        print(f"  Predictions:")
        for class_name, prob in result_dict.items():
            print(f"    {class_name}: {prob:.4f}")
        
        # Validate format
        assert isinstance(result_dict, dict)
        assert len(result_dict) == 3
        for class_name, prob in result_dict.items():
            assert isinstance(class_name, str)
            assert isinstance(prob, float)
            assert 0 <= prob <= 1
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 7: Empty text handling
    print("\n[Test 7] Testing empty text handling...")
    try:
        empty_text_embedding = classifier.embed_text("")
        print(f"✓ Empty text handled")
        print(f"  Embedding shape: {empty_text_embedding.shape}")
        
        # Should still produce valid prediction
        result = classifier.predict(
            vision_embedding=vision_embedding,
            text_embedding=empty_text_embedding,
            return_probs=False
        )
        print(f"  Prediction with empty text: {result['predicted_class']}")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 8: Get class names
    print("\n[Test 8] Testing get_class_names...")
    try:
        class_names = classifier.get_class_names()
        print(f"✓ Retrieved class names")
        print(f"  Total classes: {len(class_names)}")
        print(f"  Sample classes: {class_names[:10]}")
        assert len(class_names) > 0
        assert isinstance(class_names, list)
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "="*60)
    print("✅ ALL INFERENCE CLASSIFIER TESTS PASSED!")
    print("="*60)


if __name__ == '__main__':
    # Run vision encoder tests first and get the encoder
    vision_encoder, test_image = test_vision_encoder()
    
    # Run classifier tests using the vision encoder
    test_inference_classifier(vision_encoder, test_image)
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)