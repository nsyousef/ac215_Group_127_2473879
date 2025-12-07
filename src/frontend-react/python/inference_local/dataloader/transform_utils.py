from torchvision import transforms
from typing import Tuple, List


def get_test_valid_transform(mean: List[float], std: List[float], img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Validation transform without augmentation

    Args:
        mean: Channel-wise mean for normalization (None to disable)
        std: Channel-wise std for normalization (None to disable)
        img_size: Target image size
    """
    transform_list = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ]

    # Add normalization only if mean/std are provided
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)
