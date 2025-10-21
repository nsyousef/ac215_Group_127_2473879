import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)
from typing import Dict, Any, Optional, List

class VisionTransformer(nn.Module):
    """Vision Transformer model that returns embeddings only"""
    
    def __init__(self, vision_config: Dict[str, Any]):
        """
        Initialize Vision Transformer model
        
        Args:
            vision_config: Dictionary containing vision model configuration parameters
                Required keys:
                    - name: Model variant ('vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32')
                    - pretrained: Whether to use pretrained weights
                Optional keys:
                    - img_size: Input image size as tuple (default: (224, 224))
                    - unfreeze_layers: Number of transformer blocks to unfreeze from end (default: 0)
        """
        super().__init__()
        
        # Extract configuration parameters
        self.model_name = vision_config['name']
        self.pretrained = vision_config['pretrained']
        self.img_size = tuple(vision_config.get('img_size', (224, 224)))
        self.unfreeze_layers = vision_config.get('unfreeze_layers', 0)
        
        # Note: ViT models typically expect (224, 224) but we allow flexibility
        if self.img_size!=(224, 224):
            raise ValueError(f"Input size mismatch: {self.img_size} != (224, 224)")
        
        # Get the pretrained ViT model
        self.model = self._get_model(self.model_name, self.pretrained)
        
        self.embedding_dim = self._get_embedding_dim() # hidden dimension of the model
        
        self.model.heads = nn.Identity() # remove the classification head - we only want embeddings
        
        self._freeze_layers() # apply layer freezing/unfreezing
    
    def _get_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """Get the pretrained Vision Transformer model"""
        if model_name == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
        elif model_name == "vit_b_32":
            weights = ViT_B_32_Weights.DEFAULT if pretrained else None
            model = models.vit_b_32(weights=weights)
        elif model_name == "vit_l_16":
            weights = ViT_L_16_Weights.DEFAULT if pretrained else None
            model = models.vit_l_16(weights=weights)
        elif model_name == "vit_l_32":
            weights = ViT_L_32_Weights.DEFAULT if pretrained else None
            model = models.vit_l_32(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}. "
                           f"Supported models: vit_b_16, vit_b_32, vit_l_16, vit_l_32")
        return model
    
    def _get_embedding_dim(self) -> int:
        """Get the embedding dimension of the ViT model"""
        return self.model.hidden_dim
    
    def _freeze_layers(self, num_layers = None):
        """
        Freeze layers in the ViT model based on unfreeze_layers parameter
        unfreeze_layers: 0 means all layers frozen (except head), -1 means all unfrozen,
                        positive number means unfreeze that many transformer blocks from the end
        """
        if num_layers == -1: # This is for warmup phase, we freeze all layers
            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False
            return

        if self.unfreeze_layers == -1:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
            print("All ViT layers unfrozen")
            return
        
        # First freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.unfreeze_layers == 0:
            # Keep encoder frozen
            print("All encoder layers frozen")
            return
        
        # Unfreeze specified number of transformer blocks from the end
        encoder_blocks = list(self.model.encoder.layers.children())
        num_blocks = len(encoder_blocks)
        
        if self.unfreeze_layers > num_blocks:
            print(f"Warning: unfreeze_layers ({self.unfreeze_layers}) is greater than "
                  f"total blocks ({num_blocks}). Unfreezing all blocks.")
            blocks_to_unfreeze = encoder_blocks
        else:
            blocks_to_unfreeze = encoder_blocks[-self.unfreeze_layers:]
        
        # Unfreeze the specified blocks
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

        if self.unfreeze_layers > 0:
            for p in self.model.conv_proj.parameters():
                p.requires_grad = True
            if hasattr(self.model, "class_token"):
                self.model.class_token.requires_grad = True
            if hasattr(self.model.encoder, "pos_embedding"):
                self.model.encoder.pos_embedding.requires_grad = True
        
        # Optionally unfreeze the encoder normalization layer
        if hasattr(self.model.encoder, 'ln') and self.unfreeze_layers > 0:
            for param in self.model.encoder.ln.parameters():
                param.requires_grad = True
        
        trainable_blocks = sum(1 for block in encoder_blocks 
                              if any(p.requires_grad for p in block.parameters()))
        print(f"ViT Encoder: {num_blocks} total blocks, {trainable_blocks} unfrozen blocks")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output embeddings of shape (batch_size, embedding_dim)
        """
        # Process through conv_proj and encoder
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add class token
        batch_size = x.shape[0]
        class_token = self.model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        # Pass through encoder
        x = self.model.encoder(x)
        
        # Extract class token (embeddings)
        embeddings = x[:, 0]
        
        return embeddings
    
    def get_vision_info(self) -> Dict[str, Any]:
        """
        Get vision model information
        
        Returns:
            Dictionary containing vision model configuration and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.model.encoder.parameters() 
                               if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'embedding_dim': self.embedding_dim,
            'unfreeze_layers': self.unfreeze_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'encoder_trainable_parameters': encoder_trainable,
            'trainable_percentage': f"{(trainable_params/total_params)*100:.2f}%"
        }
        
        return info
