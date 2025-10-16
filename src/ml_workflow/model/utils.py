import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-Layer Perceptron classifier"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 activation: str = "relu", dropout_rate: float = 0.5):
        """
        Initialize MLP classifier
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output layer
            activation: Activation function ('relu', 'gelu', 'swish', 'leaky_relu')
            dropout_rate: Dropout rate between layers
        """
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build layers
        self.layers = self._build_layers()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "swish":
            return nn.SiLU()  # SiLU is the same as Swish
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _build_layers(self) -> nn.Module:
        """Build the MLP layers"""
        layers = []
        
        # Input layer
        current_size = self.input_size
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Add BatchNorm
                self._get_activation(self.activation),
                nn.Dropout(self.dropout_rate),
            ])
            current_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(current_size, self.output_size)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP"""
        return self.layers(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float or list): balancing factor for classes (scalar or per-class tensor)
            gamma (float): focusing parameter
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: raw logits (batch_size, num_classes)
            targets: ground truth labels (batch_size)
        """
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if isinstance(self.alpha, (list, torch.Tensor)):
            at = torch.tensor(self.alpha, device=inputs.device)[targets]
        else:
            at = self.alpha
        
        loss = -at * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss