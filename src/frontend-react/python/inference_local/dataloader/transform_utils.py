import torch
from torchvision import transforms
from typing import Tuple, List

def get_test_valid_transform(mean: List[float], std: List[float], img_size: Tuple[int, int]) -> transforms.Compose:
    """Validation/inference transform without augmentation"""
    return transforms.Compose([
        transforms.Resize(img_size), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])
