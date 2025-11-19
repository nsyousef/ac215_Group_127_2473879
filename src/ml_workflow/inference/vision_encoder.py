"""
Vision embedding extraction for inference.
Loads trained checkpoint and extracts embeddings from images.
"""
import torch
from PIL import Image
from typing import Union, List
import numpy as np

from ..dataloader.transform_utils import get_test_valid_transform
from ..model.vision.cnn import CNNModel
from ..utils import logger


class VisionEncoder:
    """Extracts vision embeddings using trained model."""
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = "cuda",
        img_size: tuple = (224, 224),
        model_name: str = "resnet50",
        pretrained: bool = True,
        pooling_type: str = "avg"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # Random/Pretrained initialization mode (no checkpoint)
        if checkpoint_path is None:
            logger.info(f"Initializing {model_name} (pretrained={pretrained})")
            
            vision_config = {
                'name': model_name,
                'pretrained': pretrained,
                'img_size': img_size,
                'pooling_type': pooling_type
            }
            
            self.model = CNNModel(vision_config)
            self.model.to(self.device)
            self.model.eval()
            
            # Use ImageNet defaults for normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.transform = get_test_valid_transform(mean, std, img_size)
            
            logger.info(f"Vision encoder ready on {self.device}")
            return
        
        # Checkpoint loading mode
        logger.info(f"Loading vision model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config (at top level)
        config = checkpoint.get('config', {})
        if not config:
            raise ValueError("No config found in checkpoint")
        
        vision_config = config.get('vision_model')
        if vision_config is None:
            raise ValueError(
                f"No 'vision_model' key found in checkpoint config. "
                f"Available keys: {list(config.keys())}"
            )
        
        # Create model with config from checkpoint
        logger.info(f"Creating vision model: {vision_config}")
        self.model = CNNModel(vision_config)
        
        # Load vision model weights (saved directly, no prefix)
        if 'vision_model_state_dict' not in checkpoint:
            raise ValueError(
                f"No 'vision_model_state_dict' found in checkpoint. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        
        state_dict = checkpoint['vision_model_state_dict']
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get normalization from checkpoint
        mean = checkpoint.get('normalization_mean', [0.485, 0.456, 0.406])
        std = checkpoint.get('normalization_std', [0.229, 0.224, 0.225])
        
        # Use img_size from config (not constructor param when loading checkpoint)
        self.img_size = tuple(vision_config.get('img_size', (224, 224)))
        
        self.transform = get_test_valid_transform(mean, std, self.img_size)
        
        logger.info(f"Vision encoder loaded successfully from checkpoint")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Embedding dim: {self.model.embedding_dim}")
        logger.info(f"  Image size: {self.img_size}")
        logger.info(f"  Normalization: mean={mean}, std={std}")
        logger.info(f"  Model: {vision_config['name']}")
    
    @torch.no_grad()
    def encode(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """
        Extract embedding from image.
        
        Args:
            image: Input image (PIL Image, file path, numpy array, or base64 string)
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Convert to PIL Image
            if isinstance(image, str):
                if image.startswith('data:image'):
                    # Handle base64
                    import base64
                    import io
                    header, encoded = image.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                else:
                    # File path
                    image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Preprocess and encode
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            embedding = self.model(img_tensor)
            embedding = embedding.cpu().numpy().squeeze()
            
            return embedding
            
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embedding."""
        return self.model.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'device': str(self.device),
            'embedding_dim': self.model.embedding_dim,
            'model_name': self.model.model_name,
            'pretrained': self.model.pretrained,
            'pooling_type': self.model.pooling_type,
            'img_size': self.img_size
        }
