from ml_workflow.inference.vision_encoder import VisionEncoder
from PIL import Image
import numpy as np

def test_vision_encoder():
    print("="*60)
    print("Vision Encoder Testing")
    print("="*60)

    # Test 1: Can it initialize?
    print("\n[Test 1] Initialization with pretrained ResNet50...")
    try:
        encoder = VisionEncoder(
            checkpoint_path=None,
            model_name="resnet50",
            pretrained=True,
            device="cpu"  # Use CPU for quick testing
        )
        print("✓ Initialization successful")
        print(f"  Device: {encoder.get_model_info()['device']}")
        print(f"  Embedding dim: {encoder.get_embedding_dim()}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
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
        assert embedding.shape == (2048,), f"Expected shape (2048,), got {embedding.shape}"
    except Exception as e:
        print(f"✗ FAILED: {e}")
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
        exit(1)

    print("\n" + "="*60)
    print("✅ ALL QUICK TESTS PASSED!")
    print("="*60)

if __name__ == '__main__':
    test_vision_encoder()