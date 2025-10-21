import torch
import torch.nn as nn
from typing import Dict, Any, List
from model.utils import MLP, FocalLoss

class Classifier(nn.Module):
    """Classifier module that takes embeddings and outputs class predictions"""
    
    def __init__(self, classifier_config: Dict[str, Any]):
        """
        Initialize Classifier
        
        Args:
            classifier_config: Dictionary containing classifier configuration parameters
                Required keys:
                    - input_size: Size of input embeddings (computed in main, not needed in yaml config)
                    - num_classes: Number of output classes (computed in main, not needed in yaml config)
                    - loss_fn: Loss function type ('cross_entropy' or 'focal')
                Optional keys:
                    - hidden_sizes: List of hidden layer sizes for MLP (default: [])
                    - activation: Activation function (default: 'relu')
                    - dropout_rate: Dropout rate for MLP (default: 0.0)
                    - label_smoothing: Label smoothing for cross entropy (default: 0.0)
                    - alpha, gamma, reduction: For focal loss
        """
        super().__init__()
        
        self.input_size = classifier_config['input_size']
        self.num_classes = classifier_config['num_classes']
        self.hidden_sizes = classifier_config.get('hidden_sizes', [])
        self.activation = classifier_config.get('activation', 'relu')
        self.dropout_rate = classifier_config.get('dropout_rate', 0.0)
        
        if len(self.hidden_sizes) > 0:
            self.classifier = MLP(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=self.num_classes,
                activation=self.activation,
                dropout_rate=self.dropout_rate
            )
        else:
            self.classifier = nn.Linear(self.input_size, self.num_classes)
        
        self.loss_fn = classifier_config['loss_fn']
        if self.loss_fn == 'cross_entropy':
            self.label_smoothing = classifier_config.get('label_smoothing', 0.0)
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        elif self.loss_fn == 'focal':
            if 'alpha' in classifier_config:
                self.alpha = classifier_config['alpha']
            else:
                self.alpha = classifier_config.get('sample_weights', 1.0)
            self.gamma = classifier_config.get('gamma', 2.0)
            self.reduction = classifier_config.get('reduction', 'mean')
            self.loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:

        return self.classifier(embeddings)
    
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        return self.loss_fn(outputs, targets)
    
    def get_classifier_info(self) -> Dict[str, Any]:
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': f"{(trainable_params/total_params)*100:.2f}%"
        }
