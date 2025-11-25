import torch
from torchvision import transforms
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ..utils import logger
except ImportError:
    from utils import logger


# Transform functions
def get_basic_transform(img_size: Tuple[int, int]) -> transforms.Compose:
    """Basic transform for computing statistics"""
    return transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])


def get_train_transform(
    mean: List[float], std: List[float], img_size: Tuple[int, int], augmentation_config: Dict
) -> transforms.Compose:
    """
    Training transform with augmentation

    Args:
        mean: Channel-wise mean for normalization
        std: Channel-wise std for normalization
        img_size: Target image size
        augmentation_config: Dictionary with augmentation parameters
    """
    # Extract augmentation parameters from config (use .get() with None default for optional params)
    brightness_jitter = augmentation_config.get("brightness_jitter")
    contrast_jitter = augmentation_config.get("contrast_jitter")
    saturation_jitter = augmentation_config.get("saturation_jitter")
    hue_jitter = augmentation_config.get("hue_jitter")
    rotation_degrees = augmentation_config.get("rotation_degrees")
    translate = augmentation_config.get("translate")
    scale = augmentation_config.get("scale")
    grayscale_prob = augmentation_config.get("grayscale_prob")
    horizontal_flip_prob = augmentation_config.get("horizontal_flip_prob")
    vertical_flip_prob = augmentation_config.get("vertical_flip_prob")  # Optional: defaults to None if missing

    # Build transform list dynamically based on non-zero parameters
    transform_list = []

    # Add random crop if rotation or affine transforms are used (need larger image)
    if (
        rotation_degrees is not None
        or (translate is not None and any(t > 0 for t in translate))
        or (scale is not None and any(s != 1.0 for s in scale))
    ):
        transform_list.append(transforms.Resize(int(img_size[0] * 1.1)))  # Slightly larger for random crop
        transform_list.append(transforms.RandomCrop(img_size))
    else:
        # Simple resize if no complex transforms
        transform_list.append(transforms.Resize(img_size))

    # Add flips if probability is not None
    if horizontal_flip_prob is not None:
        transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))
    if vertical_flip_prob is not None:
        transform_list.append(transforms.RandomVerticalFlip(p=vertical_flip_prob))

    # Add rotation if degrees is not None
    if rotation_degrees is not None:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    # Add affine transform if translate or scale are not None
    if (translate is not None and any(t > 0 for t in translate)) or (
        scale is not None and any(s != 1.0 for s in scale)
    ):
        transform_list.append(transforms.RandomAffine(degrees=0, translate=translate, scale=scale))

    # Add color jitter if any jitter is not None
    if (
        brightness_jitter is not None
        or contrast_jitter is not None
        or saturation_jitter is not None
        or hue_jitter is not None
    ):
        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness_jitter, contrast=contrast_jitter, saturation=saturation_jitter, hue=hue_jitter
            )
        )

    # Add grayscale if probability is not None
    if grayscale_prob is not None:
        transform_list.append(transforms.RandomGrayscale(p=grayscale_prob))

    # Always add tensor conversion and normalization
    transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    return transforms.Compose(transform_list)


def get_test_valid_transform(mean: List[float], std: List[float], img_size: Tuple[int, int]) -> transforms.Compose:
    """Validation transform without augmentation"""
    return transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )


def compute_dataset_stats(dataloader: DataLoader, num_channels: int = 3) -> Tuple[List[float], List[float]]:
    """Compute channel-wise mean and std"""
    logger.info("Computing dataset statistics...")
    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)
    total_images = 0

    for batch in tqdm(dataloader, desc="Computing stats"):
        imgs = batch[0]  # Get images
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    mean_list, std_list = mean.tolist(), std.tolist()

    logger.info(f"Mean: {mean_list}")
    logger.info(f"Std: {std_list}")

    return mean_list, std_list
