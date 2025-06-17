"""Data loaders and utilities for the cat breed classification project."""

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def print_dataset_stats(data_loaders: Dict[str, Any]) -> None:
    """
    Print statistics about the dataset with decorative elements.

    Args:
        data_loaders: Dictionary containing data loaders and class information
    """
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]
    class_names = data_loaders["class_names"]

    # Count samples per class
    train_samples = Counter()

    for _, labels in train_loader:
        for label in labels:
            train_samples[class_names[label.item()]] += 1

    # Print header
    header = "📊 Dataset Statistics 📊"
    separator = "=" * 60

    print("\n\n" + separator)
    print(f"{header:^60}")
    print(separator)

    # Print general stats
    print(f"\n🔍 {'Overview':^56} 🔍")
    print(f"{'=' * 58}")
    print(f"{'Total classes:':<30} {len(class_names):>28}")
    print(
        f"{'Total samples:':<30} "
        f"{len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset):>28}"
    )
    print(f"{'Training samples:':<30} {len(train_loader.dataset):>28}")
    print(f"{'Validation samples:':<30} {len(val_loader.dataset):>28}")
    print(f"{'Test samples:':<30} {len(test_loader.dataset):>28}")
    print(f"{'Batch size:':<30} {train_loader.batch_size:>28}")

    # Print class distribution
    print(f"\n📈 {'Class Distribution':^52} 📈")
    print(f"{'=' * 58}")

    for i, (class_name, count) in enumerate(
        sorted(train_samples.items(), key=lambda x: x[1], reverse=True)
    ):
        bar_length = int(count / max(train_samples.values()) * 20)
        bar = "█" * bar_length
        print(f"{class_name:<25} {count:>6} {bar}")

        # Print only top 10 classes if there are many
        if i >= 9 and len(class_names) > 12:
            remaining = len(class_names) - 10
            print(f"... and {remaining} more classes")
            break

    print(separator + "\n")


def print_training_config(config: Dict[str, Any]) -> None:
    """
    Print training configuration with decorative elements.

    Args:
        config: Dictionary containing training configuration
    """
    # Print header
    header = "⚙️ Training Configuration ⚙️"
    separator = "=" * 60

    print("\n" + separator)
    print(f"{header:^60}")
    print(separator)

    # Group configurations
    groups = {
        "Model": ["backbone", "pretrained", "dropout_rate", "num_classes"],
        "Optimization": ["optimizer", "learning_rate", "weight_decay", "scheduler"],
        "Training": ["epochs", "batch_size", "early_stopping", "device"],
    }

    # Print each group
    for group_name, keys in groups.items():
        print(f"\n🔧 {group_name:^56} 🔧")
        print(f"{'-' * 58}")

        for key in keys:
            if key in config:
                # Format values for better display
                value = config[key]
                if isinstance(value, bool):
                    value = "✓" if value else "✗"
                elif isinstance(value, float):
                    value = f"{value:.6f}".rstrip("0").rstrip(".")

                print(f"{key.replace('_', ' ').title():<30} {value!s:>28}")

    # Print timestamp
    print(f"\n⏱️  {'Started at:':<30} {time.strftime('%Y-%m-%d %H:%M:%S'):>28}")
    print(separator + "\n")


class CatBreedDataset(Dataset):
    """Dataset class for loading cat breed images."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize the CatBreedDataset.

        Args:
            data_dir: Directory containing class folders with images
            transform: Transforms to apply to the images
            split: One of 'train', 'val', or 'test'
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.class_names = [
            d.name
            for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        self.class_names.sort()  # Ensure consistent ordering
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        self.images, self.labels = self._load_dataset()

        # Split dataset using random seed for reproducibility
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self.images))

        test_size = int(len(indices) * test_ratio)
        val_size = int(len(indices) * val_ratio)
        train_size = len(indices) - test_size - val_size

        if split == "train":
            self.indices = indices[:train_size]
        elif split == "val":
            self.indices = indices[train_size : train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size :]

        logger.info(f"Created {split} dataset with {len(self.indices)} samples")

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """Load dataset paths and labels."""
        images = []
        labels = []

        for class_name in tqdm(self.class_names, desc="Loading dataset"):
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.glob("*.jpg"):
                images.append(img_path)
                labels.append(class_idx)

        return images, labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing the image tensor and class label
        """
        img_idx = self.indices[idx]
        img_path = self.images[img_idx]
        label = self.labels[img_idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation and testing.

    Args:
        data_dir: Directory containing class folders with images
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        img_size: Size of the input images

    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Define transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = CatBreedDataset(
        data_dir=data_dir, transform=train_transform, split="train"
    )
    val_dataset = CatBreedDataset(
        data_dir=data_dir, transform=val_test_transform, split="val"
    )
    test_dataset = CatBreedDataset(
        data_dir=data_dir, transform=val_test_transform, split="test"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    data_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": train_dataset.class_names,
        "class_to_idx": train_dataset.class_to_idx,
    }

    # Print dataset statistics
    print_dataset_stats(data_loaders)

    return data_loaders


def visualize_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    class_names: List[str],
    n_samples: int = 16,
    title: str = "Sample Batch",
) -> None:
    """
    Visualize a batch of images.

    Args:
        batch: Tuple of (images, labels)
        class_names: List of class names
        n_samples: Number of samples to display
        title: Title for the plot
    """
    images, labels = batch
    images = images[:n_samples]
    labels = labels[:n_samples]

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Plot images in a grid
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
